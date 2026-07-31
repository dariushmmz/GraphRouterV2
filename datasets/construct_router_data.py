"""
construct_router_data.py

Builds the training dataset for the LLM router.

For every question in the benchmark, each candidate LLM is asked (via
OpenRouter) to solve it. Each response is scored on:
  - Correct:         is the final answer correct? (1/0)
  - reasoning_score: how good is the step-by-step reasoning? (LLM-judged)

Token counts, latency, cost, and several diagnostic fields are also captured
so the dataset doubles as a cost/performance audit of each model.

Safety notes (since every row spends real API tokens):
  - Each row is written to disk immediately after it's computed.  A crash
    partway through loses at most the row in progress.
  - Every (question, model) pair gets a deterministic row_id.  On restart,
    rows whose row_id is already in the output CSV are skipped instead of
    re-querying the LLM -- so re-running after a crash doesn't re-spend
    tokens on already-paid-for work.
  - All progress/errors go through `logging` (console + log file) so a
    long run leaves a readable record of what happened.

Output columns
--------------
row_id, query_id, task_description, task_description_embedding,
query, query_embedding, Gold_Answer, Answer_Type, model,
Predicted_Answer, Correct, Input_Tokens, Output_Tokens, Total_Tokens,
Cost, Latency, Completion_Status, Error_Type,
Final_Answer_Length, Timestamp, reasoning_score, response
"""

import csv
import hashlib
import json
import logging
import os
import pickle
import re
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from sympy import sympify

from open_router import get_instruction, _openrouter_completion, _resolve_provider_for_model, _resolve_reasoning_for_model, _resolve_max_tokens_for_model
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

embedd_model = SentenceTransformer("all-MiniLM-L6-v2")

##
DATA_PATH        = "200Q-Final/Router_Benchmark_reduced.csv" ##
##
MODELS_JSON_PATH = "200Q-Final/LLM_Descriptions.json"
OUTPUT_PATH      = "200Q-Final/router_training_data.csv"
LOG_PATH         = "200Q-Final/router_data.log"

LLM_DESC_EMB_PATH = "200Q-Final/llm_description_embedding.pkl"

TASK_DESCRIPTION = """
The Router Benchmark is a dataset for evaluating advanced mathematical reasoning
and problem-solving capabilities.  It contains a diverse collection of
competition-level math problems (primarily from Olympiads, AIME, and similar
sources) spanning domains such as Number Theory, Combinatorics, Algebra,
Geometry, Probability, and Discrete Mathematics.  Problems range from medium to
very high difficulty and require deep conceptual understanding, multi-step
logical reasoning, symbolic manipulation, and precise computation.
The model must accurately interpret complex problem statements, apply
appropriate mathematical techniques, perform step-by-step derivations when
necessary, and produce the correct final answer (which may be a number,
expression, set, or functional form).  The primary challenges involve
high-level reasoning, constraint satisfaction, combinatorial insight, and
avoiding errors in intricate calculations.
"""

ANSWER_WEIGHT = 0.8   # weight of Correct inside a combined score (kept for
                      # backward-compatibility; not written to CSV directly)

# ---------------------------------------------------------------------------
# Load model registry from JSON
# ---------------------------------------------------------------------------

def load_model_registry(path: str) -> dict:
    """
    Load models.json and return the raw dict.

    Expected format per entry:
        {
          "DisplayName": {
            "feature":       "<description string>",
            "input_price":   <float, USD per 1M input tokens>,
            "output_price":  <float, USD per 1M output tokens>,
            "model":         "<openrouter-slug>"
          },
          ...
        }
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Load once at module level so everything downstream can reference it.
MODEL_REGISTRY: dict = load_model_registry(MODELS_JSON_PATH)

# Derived helpers -  used throughout the module
MODEL_PRICING: dict[str, tuple[float, float]] = {
    entry["model"]: (entry["input_price"], entry["output_price"])
    for entry in MODEL_REGISTRY.values()
}
CANDIDATE_MODELS: list[str] = [entry["model"] for entry in MODEL_REGISTRY.values()]

# Strong model used ONLY to grade reasoning -- never a router choice.
JUDGE_MODEL = "openai/gpt-4.1-mini"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

FIELDNAMES = [
    # Identifiers
    "row_id",
    "query_id",
    # Task
    "task_description",
    "task_description_embedding",
    # Query
    "query",
    "query_embedding",
    "Gold_Answer",
    "Answer_Type",
    # Model
    "model",
    # Response content
    "Predicted_Answer",
    "Correct",
    # Token accounting
    "Input_Tokens",
    "Output_Tokens",
    "Total_Tokens",
    # Cost
    "Cost",
    # Performance
    "Latency",
    # Diagnostics
    "Completion_Status",
    "Error_Type",
    "Final_Answer_Length",
    "Timestamp",
    # Quality
    "reasoning_score",
    # Raw output (kept for debugging / re-scoring)
    "response",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("router_data")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def savepkl(obj, path: str) -> None:
    """Persist any Python object to a pickle file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved pickle: %s", path)


def make_id(*parts: str, length: int = 12) -> str:
    """Short, deterministic id derived from one or more strings."""
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embedding(texts: List[str]) -> np.ndarray:
    """
    Return a (len(texts), dim) float32 numpy array of sentence embeddings.
    Uses the lightweight all-MiniLM-L6-v2 model by default.
    """
    return embedd_model.encode(texts, convert_to_numpy=True)


def build_and_save_llm_desc_embeddings(
    registry: dict,
    out_path: str,
) -> np.ndarray:
    """
    Embed the 'feature' description of every model in the registry and save
    the result as a pickle.

    Returns the (num_models, dim) embedding array.
    """
    descriptions = [entry["feature"] for entry in registry.values()]
    emb = get_embedding(descriptions)
    savepkl(emb, out_path)
    logger.info(
        "LLM description embeddings: shape=%s, saved to %s",
        emb.shape, out_path,
    )
    return emb


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD for a single API call."""
    if model not in MODEL_PRICING:
        logger.warning("No pricing entry for model '%s'; cost recorded as 0.", model)
        return 0.0
    price_in, price_out = MODEL_PRICING[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# ---------------------------------------------------------------------------
# Low-level LLM helper
# ---------------------------------------------------------------------------

def call_llm(
    model: str,
    prompt: str,
) -> dict:
    """
    Call OpenRouter and return a dict with keys:
        text
        input_tokens
        output_tokens
        latency
    """
    t0 = time.perf_counter()

    max_tok = _resolve_max_tokens_for_model(model)
    reasoning_kwargs = _resolve_reasoning_for_model(model)
    ignore_providers = _resolve_provider_for_model(model)

    raw = _openrouter_completion(model, prompt, max_tokens=max_tok,
                             ignore_providers=ignore_providers,
                             **reasoning_kwargs)

    latency = time.perf_counter() - t0

    # _openrouter_completion may return a plain string or a dict.
    if isinstance(raw, str):
        return {
            "text": raw,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": latency,
        }

    text = ""
    if raw.get("choices"):
        text = raw["choices"][0].get("message", {}).get("content", "")

    usage = raw.get("usage", {})

    return {
        "text": text,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "latency": latency,
    }

def parse_json_field(text: str, field: str):
    """Pull one field from an LLM JSON reply, tolerating stray text/fences."""
    cleaned = re.sub(r"```(json)?", "", text).strip()
    try:
        return json.loads(cleaned).get(field)
    except json.JSONDecodeError:
        match = re.search(rf'"{field}"\s*:\s*([0-9.]+)', cleaned)
        return float(match.group(1)) if match else None


def ensure_api_key() -> None:
    """Fail fast (before burning any tokens) if the API key isn't set."""
    if not os.getenv("OPENROUTER_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set -- check your .env file")


# ---------------------------------------------------------------------------
# Answer extraction & scoring
# ---------------------------------------------------------------------------

def extract_exact_answer(response: str) -> str:
    """Pull the answer from FINAL_ANSWER: or ### Final Answer: blocks, stripping LaTeX wrappers."""
    # Try FINAL_ANSWER: format first
    match = re.search(r"FINAL_ANSWER:\s*(.+)", response, re.DOTALL)
    if match:
        answer = match.group(1).strip().splitlines()[0].strip()
        return re.sub(r"^\\[\(\[]\s*|\s*\\[\)\]]$", "", answer).strip()

    # Try ### Final Answer: block with optional $$...$$ and \boxed{...}
    match = re.search(r"###\s*Final Answer:\s*\$\$\s*\\boxed\{(.+?)\}\s*\$\$", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: ### Final Answer: without boxed
    match = re.search(r"###\s*Final Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip().splitlines()[0].strip()
        # Strip $$ wrappers if present
        answer = re.sub(r"^\$\$\s*|\s*\$\$$", "", answer).strip()
        return re.sub(r"^\\[\(\[]\s*|\s*\\[\)\]]$", "", answer).strip()

    return ""
def _is_plain_number(s: str) -> bool:
    return bool(re.fullmatch(r"-?\s*\d+(\.\d+)?", s))


def _llm_check_equivalence(gt: str, pred: str) -> int:
    """Ask the judge LLM whether two math expressions are equivalent (1/0)."""
    prompt = f"""
    You are a math answer grader. Check if the predicted answer is mathematically equivalent to the ground truth.

    Ground Truth: {gt}
    Predicted:    {pred}

    Reply with ONLY valid JSON (nothing else):
    {{"correct": 1}} or {{"correct": 0}}
    """
    result = parse_json_field(call_llm(JUDGE_MODEL, prompt)["text"], "correct")
    return int(result) if result is not None else 0


def score_answer(gt: str, pred: str) -> int:
    """Return 1 if predicted answer matches ground truth, else 0."""
    gt   = str(gt).strip()
    pred = str(pred).strip()
    if not pred:
        return 0
    try:
        equal = sympify(gt).equals(sympify(pred))
        if equal is not None:
            return int(equal)
    except Exception:
        pass
    if _is_plain_number(pred):
        return int(gt == pred)
    return _llm_check_equivalence(gt, pred)


def llm_judge_reasoning(response: str, question: str, ground_truth: str) -> float:
    """Score the quality of the model's reasoning on a 0.0-1.0 scale."""
    prompt = f"""
You are a mathematics grader.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Student Solution:
{response}

Evaluate the mathematical reasoning using this rubric:
1.0 = Correct reasoning and correct conclusion
0.8 = Mostly correct reasoning with minor mistakes
0.6 = Significant progress toward solution
0.4 = Some relevant mathematical steps
0.2 = Minimal useful reasoning
0.0 = Irrelevant or incorrect reasoning

Return ONLY JSON, no extra text, in exactly this format:
{{"reasoning_score": 1.0}}
"""
    result = parse_json_field(call_llm(JUDGE_MODEL, prompt)["text"], "reasoning_score")
    return float(result) if result is not None else 0.0


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_error(
    response: str,
    predicted_answer: str,
    correct: int,
    latency: float,
    completion_status: str,
) -> str:
    """
    Categories
    ----------
    none            – answered correctly; no error
    wrong_answer    – answer present but incorrect
    no_final_answer – responded but no FINAL_ANSWER: tag found
    timeout         – completion_status flagged a timeout
    api_error       – completion_status flagged an API/network failure
    empty_response  – response was blank
    """
    if completion_status == "timeout":
        return "timeout"
    if completion_status == "api_error":
        return "api_error"
    if not response.strip():
        return "empty_response"
    if not predicted_answer:
        return "no_final_answer"
    if correct:
        return "none"
    return "wrong_answer"


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(
    question:                   str,
    gold_answer:                str,
    answer_type:                str,
    model:                      str,
    row_id:                     str,
    question_id:                str,
    task_desc:        str,
) -> dict:
    """
    Query one model on one question and assemble a full result row.
    Never raises: all exceptions are caught and reflected in status fields.
    """
    timestamp         = datetime.now(timezone.utc).isoformat()
    response          = ""
    predicted_answer  = ""
    correct           = 0
    input_tokens      = 0
    output_tokens     = 0
    latency           = 0.0
    reasoning_score   = 0.0
    completion_status = "success"

    # Embed the query (done once per question across all models via the caller,
    # but kept here for simplicity; no extra cost since it's local inference).
    query_embedding = get_embedding([question])
    task_desc_embedding = get_embedding([task_desc])

    try:
        prompt    = get_instruction(question)
        call_info = call_llm(model, prompt)
        response      = call_info["text"]
        input_tokens  = call_info["input_tokens"]
        output_tokens = call_info["output_tokens"]
        latency       = round(call_info["latency"], 3)

        if not isinstance(response, str):
            logger.error(
                "Unexpected response type: %s, value=%r",
                type(response).__name__,
                response
            )
            completion_status = "api_error"

        elif not response.strip():
            completion_status = "api_error"

        else:
            predicted_answer = extract_exact_answer(response)
            correct = score_answer(gold_answer, predicted_answer)
            reasoning_score = None
            # reasoning_score  = llm_judge_reasoning(response, question, gold_answer)


    except TimeoutError:
        completion_status = "timeout"
        logger.exception("row %s timed out (model=%s)", row_id, model)
    except Exception:
        completion_status = "api_error"
        logger.exception("row %s failed (model=%s)", row_id, model)

    error_type          = classify_error(response, predicted_answer, correct, latency, completion_status)
    total_tokens        = input_tokens + output_tokens
    cost                = compute_cost(model, input_tokens, output_tokens)
    final_answer_length = len(predicted_answer)

    return {
        "row_id":                     row_id,
        "query_id":                   question_id,
        "task_description":           task_desc,
        "task_description_embedding": task_desc_embedding.tolist(),
        "query":                      question,
        "query_embedding":            query_embedding.tolist(),
        "Gold_Answer":                gold_answer,
        "Answer_Type":                answer_type,
        "model":                      model,
        "Predicted_Answer":           predicted_answer,
        "Correct":                    correct,
        "Input_Tokens":               input_tokens,
        "Output_Tokens":              output_tokens,
        "Total_Tokens":               total_tokens,
        "Cost":                       round(cost, 8),
        "Latency":                    latency,
        "Completion_Status":          completion_status,
        "Error_Type":                 error_type,
        "Final_Answer_Length":        final_answer_length,
        "Timestamp":                  timestamp,
        "reasoning_score":            reasoning_score,
        "response":                   response,
    }


# ---------------------------------------------------------------------------
# Incremental CSV writing / resuming
# ---------------------------------------------------------------------------

def load_completed_row_ids(output_path: str) -> set:
    """Return the set of row_ids already saved from a previous (crashed) run."""
    if not os.path.exists(output_path):
        return set()
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["row_id"] for row in reader if row.get("row_id")}


def build_router_dataset(
    questions:            list[str],
    answers:              list[str],
    answer_types:         list[str],
    task_descs:  list[str],
    models:               list[str] = None,
    output_path:          str       = OUTPUT_PATH,
) -> None:
    """
    Stream one row per (question, model) pair straight to `output_path`.

    Safe to re-run: rows already present in `output_path` are skipped,
    so a crashed run can simply be restarted without re-spending tokens.
    """
    if models is None:
        models = CANDIDATE_MODELS

    completed   = load_completed_row_ids(output_path)
    file_exists = os.path.exists(output_path)
    total       = len(questions) * len(models)

    logger.info(
        "Starting run: %d questions x %d models = %d rows (%d already done)",
        len(questions), len(models), total, len(completed),
    )

    done = failed = skipped = 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for question, gold_answer, answer_type, task_desc in zip(questions, answers, answer_types, task_descs):
            question_id = make_id(question)

            for model in models:

                # ==================
                # if model in ['openai/gpt-5', 'deepseek/deepseek-r1']:
                #     continue
                if model != 'deepseek/deepseek-r1':
                # ==================


                    row_id = make_id(model, question)

                    if row_id in completed:
                        skipped += 1
                        continue

                    seq = done + failed + skipped + 1
                    logger.info(
                        "[%d/%d] model=%s  query_id=%s  -> %.60s",
                        seq, total, model, question_id, question,
                    )

                    row = process_row(
                        question, gold_answer, answer_type,
                        model, row_id, question_id,
                        task_desc,
                    )

                    writer.writerow(row)
                    f.flush()   # persist immediately -- don't lose a paid-for row
                    completed.add(row_id)

                    if row["Completion_Status"] == "success":
                        done += 1
                    else:
                        failed += 1

    logger.info(
        "Finished: %d done, %d failed, %d skipped. Saved to %s",
        done, failed, skipped, output_path,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_api_key()

    # ── 1. Embed LLM descriptions (once; cheap local inference) ─────────────
    llm_desc_emb = build_and_save_llm_desc_embeddings(
        registry=MODEL_REGISTRY,
        out_path=LLM_DESC_EMB_PATH,
    )

    # ── 2. Embed the shared task description (once for the whole run) ────────
    task_desc_embedding = get_embedding([TASK_DESCRIPTION])  # shape (1, dim)

    # ── 3. Load benchmark ────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, nrows=100)

    # targeted = r"""Suppose that we are given 40 points equally spaced around the perimeter of a square, so that four of them are located at the vertices and the remaining points divide each side into ten congruent segments.  If $P$, $Q$, and $R$ are chosen to be any three of these points which are not collinear, then how many different possible positions are there for the centroid of $\triangle PQR$?"""
    # df  = df[df['question']==targeted]



    answer_types = (
        df["answer_type"].tolist()
        if "answer_type" in df.columns
        else ["latex"] * len(df)
    )

    # ── 4. Build the router training dataset ─────────────────────────────────
    build_router_dataset(
        questions           = df["question"].tolist(),
        answers             = df["answer"].tolist(),
        answer_types        = answer_types,
        task_descs = df['reasoning_type'].tolist(),
    )

    # ── 5. Build semantic query embeddings from the finished CSV ─────────────
    # import torch
    # from utils import build_semantic_embedd
    #
    # root_dir   = DATA_PATH.split("/")[0]
    # router_csv = os.path.join(root_dir, "router_training_data.csv")
    # emb_out    = os.path.join(root_dir, "query_semantic_embeddings.pkl")
    # device     = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # build_semantic_embedd(
    #     router_csv = router_csv,
    #     num_llms   = len(CANDIDATE_MODELS),
    #     model_name = "paraphrase-multilingual-MiniLM-L12-v2",
    #     batch_size = 64,
    #     out_path   = emb_out,
    #     device     = device,
    # )