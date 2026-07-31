import json
import re
import string
import pickle

import numpy as np
import litellm
import hashlib
import os
import requests
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Union



# File I/O functions
def loadjson(filename: str) -> dict:
    """
    Load data from a JSON file.

    Args:
        filename: Path to the JSON file

    Returns:
        Dictionary containing the loaded JSON data
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def savejson(data: dict, filename: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        filename: Path where the JSON file will be saved
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def loadpkl(filename: str) -> any:
    """
    Load data from a pickle file.

    Args:
        filename: Path to the pickle file

    Returns:
        The unpickled object
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def savepkl(data: any, filename: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data: Object to save
        filename: Path where the pickle file will be saved
    """
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def save_feedback(
    query_id: int,
    llm_name: str,
    user_score: float,
    feedback_path: str,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save user feedback to a file.
    
    Args:
        query_id: Query identifier
        llm_name: Name of the LLM
        user_score: User-provided score
        feedback_path: Path to feedback file
        extra: Optional additional metadata
    """
    record = {
        "timestamp": time.time(),
        "query_id": int(query_id),
        "LLM": llm_name,
        "Score": float(user_score),
    }
    
    if extra:
        record.update(extra)
    
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ask_and_save_feedback(
    query_id: int,
    predicted_llm: str,
    feedback_path: str
) -> None:
    """
    Prompt user for feedback and save it.
    
    Args:
        query_id: Query identifier
        predicted_llm: Name of predicted LLM
        feedback_path: Path to feedback file
    """
    user_input = input("\n\n===========> Score 1..5 (or blank to reject): ").strip()
    if user_input.isdigit():
        score = int(user_input)
        save_feedback(query_id, predicted_llm, score, feedback_path)
    return score



# Text normalization and evaluation functions
def normalize_answer(s: str, normal_method: str = "") -> str:
    """
    Normalize text for evaluation.

    Args:
        s: String to normalize
        normal_method: Method for normalization ("mc" for multiple choice, "" for standard)

    Returns:
        Normalized string
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def mc_remove(text):
        a1 = re.findall(r'\([a-zA-Z]\)', text)
        if len(a1) == 0:
            return ""
        return a1[-1]


    if normal_method == "mc":
        return mc_remove(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Calculate F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str, normal_method: str = "") -> bool:
    """
    Check if prediction exactly matches ground truth after normalization.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        normal_method: Method for normalization

    Returns:
        True if exact match, False otherwise
    """
    return (normalize_answer(prediction, normal_method=normal_method) ==
            normalize_answer(ground_truth, normal_method=normal_method))


def get_bert_score(generate_response: List[str], ground_truth: List[str]) -> float:
    from bert_score import score

    """
    Calculate BERT score between generated responses and ground truths.

    Args:
        generate_response: List of generated responses
        ground_truth: List of ground truth texts

    Returns:
        Average BERT score (F1)
    """
    F_l = []
    for inter in range(len(generate_response)):
        generation = generate_response[inter]
        gt = ground_truth[inter]
        P, R, F = score([generation], [gt], lang="en", verbose=True)
        F_l.append(F.mean().numpy().reshape(1)[0])
    return np.array(F_l).mean()


# Embedding and dimensionality reduction
def reduce_embedding_dim(embed: np.ndarray, dim: int = 50) -> np.ndarray:
    from sklearn.decomposition import PCA

    """
    Reduce dimensionality of embeddings using PCA.

    Args:
        embed: Embedding vectors
        dim: Target dimension

    Returns:
        Reduced embeddings
    """
    pca = PCA(n_components=dim)
    reduced_embeddings = pca.fit_transform(embed)
    return reduced_embeddings


def get_embedding(instructions: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    """
    Get embeddings for a list of texts and optionally reduce dimensions.

    Args:
        instructions: List of texts to embed
        dim: Target dimension for embeddings

    Returns:
        Numpy array of embeddings
    """
    emb_list = model.encode(instructions)
    return emb_list


import requests
import json
import time
from typing import Optional

def _openrouter_completion(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    retries: int = 3,
    retry_delay: float = 3.0,
) -> str:
    from dotenv import load_dotenv
    load_dotenv() 

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"




    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "LLM Selection Project",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    if top_p is not None:
        payload["top_p"] = top_p

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    # ---------- Validate response ----------
    if (
        "choices" not in data
        or not data["choices"]
        or "message" not in data["choices"][0]
        or "content" not in data["choices"][0]["message"]
    ):
        raise ValueError("Invalid OpenRouter response structure")

    content = data["choices"][0]["message"]["content"]

    # Empty or nonsense response
    if not content or len(content.strip()) < 5:
        raise ValueError("Empty or incomplete LLM response")

    return content


def model_prompting(
    llm_model: str,
    prompt: str,
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
) -> str:
    """
    Get a response from an LLM model.
    Primary: LiteLLM
    Fallback: OpenRouter
    """

    # -------- Primary: LiteLLM --------
    if not llm_model in ['google/gemini-2.5-pro', 'openai/gpt-4.1']:
        try:
            completion = litellm.completion(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                n=return_num,
                top_p=top_p,
                max_tokens=max_token_num,
                temperature=temperature,
                stream=stream,
            )
            return completion.choices[0].message.content

        except Exception as e:
            # You can narrow this to specific LiteLLM exceptions if desired
            print(f"[LiteLLM failed] {e}")

    else:
        # -------- Fallback: OpenRouter --------
        try:
            return _openrouter_completion(
                model=llm_model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
            )

        except Exception as e:
            raise RuntimeError(
                f"\nBoth LiteLLM and OpenRouter failed for model '{llm_model}'. "
                f"Last error: {e}"
            )


# ======================================================
# Simple logger helper
# ======================================================

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


# ======================================================
# Utility functions
# ======================================================

def extract_final_answer(answer_text):
    if answer_text and "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return None


def extract_reasoning(answer_text):
    if answer_text and "####" in answer_text:
        return answer_text.split("####")[0].strip()
    return answer_text


import re

def normalize_text(s):
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = s.lower().strip()
    return s

from sympy import sympify, simplify
from sympy.parsing.latex import parse_latex

def strip_latex_math(expr: str) -> str:
    if not isinstance(expr, str):
        return expr

    expr = expr.strip()

    # Remove \( ... \)
    expr = re.sub(r'^\\\((.*)\\\)$', r'\1', expr)

    # Remove \[ ... \]
    expr = re.sub(r'^\\\[(.*)\\\]$', r'\1', expr)

    # Remove $ ... $
    expr = re.sub(r'^\$(.*)\$$', r'\1', expr)

    return expr.strip()

def parse_expr(expr):
    expr = strip_latex_math(expr)

    # Try sympify first (handles 1/2, sqrt(2)/3, etc.)
    try:
        return sympify(expr)
    except Exception:
        pass

    # Fallback to LaTeX
    try:
        return parse_latex(expr)
    except Exception:
        raise ValueError(f"Could not parse expression: {expr}")

from sympy import simplify

def math_equal(expr1, expr2):
    try:
        e1 = parse_expr(expr1)
        e2 = parse_expr(expr2)
        return simplify(e1 - e2) == 0
    except Exception as e:
        # optional: log instead of print in production
        print("math_equal error:", e)
        return False


import sympy as sp

def numeric_equal(expr1, expr2, tol=1e-6):
    try:
        e1 = float(sp.N(parse_expr(expr1)))
        e2 = float(sp.N(parse_expr(expr2)))
        return abs(e1 - e2) < tol
    except Exception:
        return False




def answer_match(ans1, ans2):
    if ans1 is None or ans2 is None:
        return 0

    # 1. Normalized text match
    if normalize_text(ans1) == normalize_text(ans2):
        return 1

    # 2. Symbolic equivalence
    if math_equal(ans1, ans2):
        return 1

    # 3. Numeric fallback
    if numeric_equal(ans1, ans2):
        return 1

    return 0


def extract_final_answer_from_response(response_text):
    if response_text is None:
        return None

    # text = response_text.replace("\n", "").strip()
    text = response_text

    answer_part = None
    answer_patterns = [
        r"A:\s*(.*)$",
        r"Answer:\s*(.*)$"
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            break


    if answer_part is None:
        answer_part = text.split(".")[-1].strip()

    answer_part = re.split(
        r"(therefore|thus|hence|so|which means)",
        answer_part,
        flags=re.IGNORECASE
    )[0].strip()

    boxed_match = re.search(r"\\boxed\s*\{(.+?)\}", answer_part)
    if boxed_match:
        return boxed_match.group(1).strip()

    if "=" in answer_part:
        rhs = answer_part.split("=")[-1].strip()
        if rhs:
            answer_part = rhs

    if answer_part.startswith("$") and answer_part.endswith("$"):
        answer_part = answer_part[1:-1].strip()

    return answer_part if answer_part else None


def is_numeric_ground_truth(gt):
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", str(gt).strip()))


def normalize_numeric_answer(text):
    if text is None:
        return None

    text = str(text)
    text = re.sub(r"\^\\circ", "", text)
    text = re.sub(r"[°]", "", text)
    text = re.sub(r"(degrees?|deg)", "", text, flags=re.I)
    text = text.replace("\\,", "").replace("$", "").strip()

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None

def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

def build_uid(task_id, query, llm):
    q_hash = stable_hash(query)
    return f"{task_id}::{q_hash}::{llm}", q_hash

    
def parse_embedding(raw: Union[str, List, np.ndarray]) -> np.ndarray:
    """
    Extract float values from a string representation of an embedding.
    
    Args:
        raw: Raw embedding data (string, list, or numpy array)
        
    Returns:
        Numpy array of float values
    """
    if isinstance(raw, str):
        # Extract all float-like tokens from the string
        pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?'
        nums = re.findall(pattern, raw)
        return np.array([float(n) for n in nums], dtype=float)
    
    return np.array(raw, dtype=float)

def parse_embedding_field(raw: Union[str, List, np.ndarray]) -> np.ndarray:
    """
    Parse an embedding field that may be in various formats.
    
    Args:
        raw: Raw embedding data in various formats
        
    Returns:
        Numpy array of float values
        
    Raises:
        ValueError: If the embedding cannot be parsed
    """
    if isinstance(raw, (list, np.ndarray)):
        return np.array(raw, dtype=float)
    
    try:
        return parse_embedding(raw)
    except (ValueError, TypeError):
        # Try to parse as JSON string
        s = str(raw).strip()
        s = re.sub(r'\s+', ', ', s)
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            # Try fixing common JSON formatting issues
            parsed = json.loads(s.replace("[[,", "[["))
        
        if isinstance(parsed, list) and len(parsed) > 0:
            return np.array(parsed[0] if isinstance(parsed[0], list) else parsed, dtype=float)
        return np.array(parsed, dtype=float)


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure an array is 2-dimensional.
    
    Args:
        arr: Input array (1D or 2D)
        
    Returns:
        2D array (reshaped if necessary)
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


import plotly.io as pio
import plotly.express as px
import pandas as pd, os

def make_plot(scores, output_dir, scenario, query_id):
    results_list = []
    for llm, score in scores.items():
        results_list.append({
            "LLM": llm,
            "Score": score,
            "Scenario": scenario
        })

    df_scores = pd.DataFrame(results_list)
    # Compute min/max for y-axis with margin
    y_min = df_scores["Score"].min()
    y_max = df_scores["Score"].max()
    margin = (y_max - y_min) * 0.05  # 5% margin
    
    # Create interactive grouped bar chart
    fig = px.bar(
        df_scores,
        x="LLM",
        y="Score",
        color="Scenario",
        barmode="group",
        text=df_scores["Score"].apply(lambda x: f"{x:.3f}"),
        title=f"LLM Scores Across Different Scenarios for Query {query_id}"
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="LLM",
        yaxis_title="Score",
        xaxis_tickangle=-45,
        yaxis=dict(showgrid=True, range=[y_min - margin, y_max + margin]),
        legend_title="Scenario",
        template="plotly_white"
    )
    
    # Save as HTML file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"llm_scores_query_{query_id}.html")
    fig.write_html(output_file)
    print(f"Graph saved as: {output_file}")

    # pio.renderers.default = "browser"
    fig.show()