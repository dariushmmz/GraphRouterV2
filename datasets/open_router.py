import json
import os
import time
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

instruction = r"""Solve the math problem below. Output ONLY this exact format — nothing else:

FINAL_ANSWER:
[answer]

Rules:
- Be concise. No reasoning, steps, or explanation.
- LaTeX for math expressions (e.g. \\frac{m}{n}, \\sqrt{2}).
- Plain numbers for integers/decimals.
- Do not prove the answer.
- Do not verify the answer.
- Do not check alternative cases.

Problem: """


def get_instruction(question: str) -> str:
    return instruction+question


# Reasoning config builder
def _reasoning_config(
    max_reasoning_tokens: Optional[int] = None,
    effort: Optional[str] = None,
    exclude: bool = False,
) -> Optional[dict]:
    """
    Build the `reasoning` payload for OpenRouter.

    Priority:
      1. max_reasoning_tokens — direct token budget (Anthropic, Qwen, Gemini 2.5)
      2. effort              — level-based ("none"/"minimal"/"low"/"medium"/"high"/"max")
      3. If both are None and exclude=False → omit the key entirely (model default)

    NOTE: max_tokens and effort are mutually exclusive in OpenRouter's API.
    Sending both in the same `reasoning` object causes a 400 Bad Request.
    If both are provided, max_reasoning_tokens takes priority and effort is ignored.
    """
    if max_reasoning_tokens is None and effort is None and not exclude:
        return None  # Don't send the key at all — model decides

    config: dict = {}
    if max_reasoning_tokens is not None:
        config["max_tokens"] = max_reasoning_tokens
    elif effort is not None:
        config["effort"] = effort
    if exclude:
        config["exclude"] = True

    return config


def _resolve_max_tokens_for_model(model: str) -> int|None:
    """
    Total completion token ceiling per model, sized from observed
    output-token distributions in the router benchmark dataset.
    Must be generous enough to cover reasoning + FINAL_ANSWER: text,
    or extraction silently fails (no_final_answer / api_error length).
    """
    m = model.lower()

    if "deepseek" in m and "r1" in m:
        # return 32000          # p90 ~18.5k, max seen ~29.7k -> headroom
        return None
    if "gpt-5" in m:
        return 48000          # p90 ~12.7k, max seen ~43.6k -> headroom
    if "gemini" in m:
        return 70000          # one outlier hit 64k
    if "claude" in m and "sonnet" in m:
        return 4000           # p90 ~1.7k, current cap is fine
    if "qwen" in m and "thinking" in m:
        return 4000           # p90 ~1.2k, current cap is fine
    if "llama" in m or "mixtral" in m:
        return 512            # trivial outputs, no reasoning

    return 8000  # safe default for anything unlisted

def _resolve_provider_for_model(model: str) -> Optional[list]:
    m = model.lower()
    if "deepseek" in m and "r1" in m:
        return ["Azure"]   # Azure's R1 deployment is broken/unavailable
    return None

# ---------------------------------------------------------------------------
# Per-model reasoning policy
# ---------------------------------------------------------------------------

def _resolve_reasoning_for_model(model: str) -> dict:
    """
    Return the reasoning kwargs to use for a given OpenRouter model slug,
    based on each model's actual supported control mode and our cost goals.

    - GPT-5            -> effort-based, well-documented % budgets. Use "low".
    - Claude-Sonnet-4.5 -> Anthropic uses max_tokens (extended thinking budget).
                           Use a modest hard cap.
    - Gemini-3.5-Flash  -> effort-based (thinkingLevel). Use "low".
    - DeepSeek-R1       -> only reliably respects max_tokens. Use a hard cap.
    - Qwen3-235B-Thinking -> hybrid thinking model, supports max_tokens budget.
                              Use a hard cap.
    - Llama-3.1-70B     -> not a reasoning model. No reasoning field needed.
    - Mixtral-8x22B     -> not a reasoning model. No reasoning field needed.

    Returns a dict of kwargs to splat into _openrouter_completion:
        {"reasoning_effort": ..., "max_reasoning_tokens": ..., "exclude_reasoning": ...}
    Only one of reasoning_effort / max_reasoning_tokens is ever set (they're
    mutually exclusive on OpenRouter's API).
    """
    m = model.lower()

    if "deepseek" in m and "r1" in m:
        # return { "reasoning_effort": "low",
        #     "exclude_reasoning": True}  # no effort/max_tokens cap at all — let it think as long as it needs, just don't stream the trace back
        return {}
    if "gpt-5" in m:
        return {"reasoning_effort": "medium", "exclude_reasoning": True}

    # --- Non-reasoning models: don't send a reasoning field at all ---
    if "llama" in m or "mixtral" in m:
        return {}

    if "gemini" in m:
        return {"reasoning_effort": "low", "exclude_reasoning": True}

    # --- max_tokens-based models ---
    if "claude" in m and "sonnet" in m:
        return {"max_reasoning_tokens": 4000, "exclude_reasoning": True}

    if "qwen" in m and "thinking" in m:
        return {"max_reasoning_tokens": 4000, "exclude_reasoning": True}

    # --- default fallback: let the model/OpenRouter decide ---
    return {}

# OpenRouter completion
def _openrouter_completion(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    # --- Reasoning token control ---
    # Set effort="none" to fully disable reasoning on thinking models.
    # Set max_reasoning_tokens=1024 for a cheap hard cap.
    # Both default to None → OpenRouter/model decides.
    reasoning_effort: Optional[str] = None,
    max_reasoning_tokens: Optional[int] = None,
    exclude_reasoning: bool = False,
    ignore_providers: Optional[list] = None,
) -> dict:
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — check your .env file")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "LLM Router Benchmark",
    }

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens


    if top_p is not None:
        payload["top_p"] = top_p

    reasoning = _reasoning_config(
        max_reasoning_tokens=max_reasoning_tokens,
        effort=reasoning_effort,
        exclude=exclude_reasoning,
    )
    if reasoning is not None:
        payload["reasoning"] = reasoning

    if ignore_providers:
        payload["provider"] = {"ignore": ignore_providers}

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()

    if (
        "choices" not in data
        or not data["choices"]
        or "message" not in data["choices"][0]
        or "content" not in data["choices"][0]["message"]
    ):
        raise ValueError(f"Unexpected OpenRouter response structure: {data}")

    return data