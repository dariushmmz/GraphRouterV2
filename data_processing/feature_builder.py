"""
feature_builder.py

Implements Revised_Feature_Plan.md Phases 2/3, 4, 6 for GraphRouterV2.

Responsibilities
-----------------
1. Join router_data.csv (per Question-Model record) with the benchmark file
   (per-Question static metadata) on query <-> question.
2. Build Phase 2/3 question-node metadata vector:
   [Domain, Reasoning_Type, Difficulty, Reasoning_Depth, Solution_Steps,
    Answer_Type, Question_Length]
   The final node feature fed to the model is
   concat(query_embedding, metadata_vector) -- the "Concat" half of
   Question Representation = MLP(Concat(...)); the MLP itself is
   FeatureAlign.query_transform inside graph_nn.py (unchanged).

   CONFIRMED (from real data): `difficulty` and `reasoning_depth` are NOT
   clean numeric/ordinal columns. Observed values:
     difficulty:       '7','8','9','Medium','Medium-Hard','Hard','High',
                        'Very High','Very High / Top','Easy-Medium',
                        'Very Hard','5','6'
     reasoning_depth:   '4','5','Medium-High','High','Medium','Very High','3'
   Both mix numeric-string values with text labels on an unspecified
   combined scale -- there is no evident single ordering across the mix
   (is '7' below or above 'Medium-Hard'?). Both are therefore treated as
   CATEGORICAL (one-hot), each distinct string its own category, by
   default. Pass `difficulty_ordinal_order=[...]` /
   `reasoning_depth_ordinal_order=[...]` to `fit_encoders()` if you later
   determine the real combined ordering (e.g. if the numeric and text
   values come from different `source` values in the benchmark and can be
   bucketed onto one scale).
3. Build Phase 4 edge feature vector:
   [Correct, Cost_norm, Latency_norm, Input_Tokens_norm, Output_Tokens_norm,
    Completion_Status_onehot, Error_Type_onehot]
4. Build a Phase 6 utility score per (query, model) edge, used both as the
   training label (argmax per query -> one-hot) and as the ground-truth
   ranking signal at inference time, replacing the old effect/cost-only
   scenario logic:

       Utility = w1*Correct - w2*Cost_norm - w3*Latency_norm
                 - w4*Output_Tokens_norm + w5*Completion_Reliability

   reasoning_score and User_Feedback terms are intentionally omitted
   (NaN / not collected -- see Revised_Feature_Plan.md).

Fitting discipline (train-only, no leakage)
--------------------------------------------
`fit_encoders()` MUST be called on the TRAIN-SPLIT rows only. All the
`build_*`/`compute_utility` functions are pure transforms that take
(any_dataframe, already-fitted encoders) and apply the train-fit
categories/normalization stats to it -- so the correct usage is:

    train_df = full_df.iloc[train_row_idx]
    enc = fit_encoders(train_df, ...)
    node_meta   = build_node_metadata(full_df, enc)   # applied to ALL rows
    edge_feats  = build_edge_features(full_df, enc)   # applied to ALL rows
    utility     = compute_utility(full_df, enc)        # applied to ALL rows

`multi_task_graph_router.py` now computes the train/val/test row indices
BEFORE fitting encoders (moved out of split_data() into a standalone
`_compute_split_row_indices()` called early in `__init__`), so val/test
rows never influence one-hot category discovery or z-score mu/sigma.

Known limitations still open
-------------------------------
* Join key is `Gold_Answer` (router_data) <-> `answer` (benchmark), NOT
  question text. This is a WEAKER key than a real question ID: distinct
  questions can share an answer string (e.g. two problems both answering
  "42"). `join_router_and_benchmark()` reports
  `JoinReport.n_ambiguous_benchmark_keys` -- if non-zero, some rows are
  getting another question's metadata. Get a real question ID column if
  one exists before trusting node metadata in that case.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Sequence


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _normalize_str(s) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _zscore_fit(x: np.ndarray):
    mu, sigma = np.nanmean(x), np.nanstd(x)
    sigma = sigma if sigma > 1e-12 else 1.0
    return mu, sigma


def _zscore_apply(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (np.nan_to_num(x, nan=mu) - mu) / sigma


@dataclass
class JoinReport:
    n_router_rows: int
    n_matched: int
    n_ambiguous_benchmark_keys: int = 0  # answer values that map to >1 distinct question in benchmark

    @property
    def match_rate(self) -> float:
        return self.n_matched / max(self.n_router_rows, 1)


# --------------------------------------------------------------------------
# Join
# --------------------------------------------------------------------------

def join_router_and_benchmark(
    router_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    router_key_col: str = "query",
    benchmark_key_col: str = "question",
) -> tuple[pd.DataFrame, JoinReport]:
    """
    Left-join router_data onto benchmark metadata via router_data.query
    <-> benchmark.question (normalized exact string match).
    """
    router_df = router_df.copy()
    benchmark_df = benchmark_df.copy()

    router_df["_join_key"] = router_df[router_key_col].map(_normalize_str)
    benchmark_df["_join_key"] = benchmark_df[benchmark_key_col].map(_normalize_str)

    # Detect collisions: benchmark rows sharing the same normalized key
    # (e.g. two rows with identical question text) -- first occurrence wins,
    # rest are dropped; report how many keys this affected.
    key_counts = benchmark_df.groupby("_join_key").size()
    n_ambiguous = int((key_counts > 1).sum())

    benchmark_dedup = benchmark_df.drop_duplicates(subset="_join_key", keep="first")

    merged = router_df.merge(
        benchmark_dedup,
        on="_join_key",
        how="left",
        suffixes=("", "_bench"),
    )

    n_matched = merged["_join_key"].isin(benchmark_dedup["_join_key"]).sum()
    report = JoinReport(
        n_router_rows=len(router_df),
        n_matched=int(n_matched),
        n_ambiguous_benchmark_keys=n_ambiguous,
    )
    return merged, report


# --------------------------------------------------------------------------
# Encoder container (fit once, reuse at inference time)
# --------------------------------------------------------------------------

@dataclass
class FeatureEncoders:
    # node metadata
    domain_categories: list = field(default_factory=list)
    domain_precomputed_columns: list = field(default_factory=list)
    reasoning_type_categories: list = field(default_factory=list)
    difficulty_categories: list = field(default_factory=list)
    difficulty_ordinal_order: Optional[list] = None
    difficulty_is_numeric: bool = False
    difficulty_mu_sigma: tuple = (0.0, 1.0)
    # NOTE: reasoning_depth is NOT purely numeric in the real benchmark --
    # observed values mix numeric strings ('3','4','5') with text labels
    # ('Medium','Medium-High','High','Very High') on an unspecified combined
    # scale. Treated as categorical (one-hot) like difficulty, not z-scored.
    reasoning_depth_categories: list = field(default_factory=list)
    reasoning_depth_ordinal_order: Optional[list] = None
    reasoning_depth_is_numeric: bool = False
    reasoning_depth_mu_sigma: tuple = (0.0, 1.0)
    answer_type_categories: list = field(default_factory=list)
    solution_steps_mu_sigma: tuple = (0.0, 1.0)
    question_length_mu_sigma: tuple = (0.0, 1.0)

    # edge features
    completion_status_categories: list = field(default_factory=list)
    error_type_categories: list = field(default_factory=list)
    cost_mu_sigma: tuple = (0.0, 1.0)
    latency_mu_sigma: tuple = (0.0, 1.0)
    input_tokens_mu_sigma: tuple = (0.0, 1.0)
    output_tokens_mu_sigma: tuple = (0.0, 1.0)

    # completion reliability per model, computed at fit time
    completion_reliability_by_model: dict = field(default_factory=dict)

    @property
    def node_metadata_dim(self) -> int:
        # NOTE: domain macro-category is NOT included here -- it moved off
        # the Question node onto the shared Reasoning prototype nodes (see
        # reasoning_node_dim / build_reasoning_node_features /
        # assign_reasoning_node_index below). Everything else
        # (reasoning_type, difficulty, reasoning_depth, answer_type,
        # solution_steps, question_length) stays on the Question node.
        diff_dim = 1 if (self.difficulty_ordinal_order or self.difficulty_is_numeric) else len(self.difficulty_categories)
        rdepth_dim = 1 if (self.reasoning_depth_ordinal_order or self.reasoning_depth_is_numeric) else len(self.reasoning_depth_categories)
        return (
            len(self.reasoning_type_categories)
            + diff_dim
            + rdepth_dim
            + 1  # solution_steps
            + len(self.answer_type_categories)
            + 1  # question_length
        )

    @property
    def reasoning_categories(self) -> list:
        """
        Ordered list of the domain macro-category names in use, one per
        shared Reasoning prototype node. Mirrors the domain
        precomputed-columns-vs-bucketed-text fallback used in
        build_node_metadata previously: prefers the fixed
        DOMAIN_MACRO_CATEGORIES order (derived from whichever source was
        fit), so the Reasoning node count/order is stable across
        train/val/test splits and across fit_encoders() calls.
        """
        if self.domain_precomputed_columns:
            return [c[len("domain_"):] for c in self.domain_precomputed_columns]
        return list(self.domain_categories)

    @property
    def reasoning_node_dim(self) -> int:
        """
        Feature width of each Reasoning prototype node = number of domain
        macro-categories. Each prototype node's own input feature is its
        one-hot identity within this set (see build_reasoning_node_features).
        """
        return len(self.reasoning_categories)

    @property
    def edge_feature_dim(self) -> int:
        return (
            1  # Correct
            + 1  # Cost_norm
            + 1  # Latency_norm
            + 1  # Input_Tokens_norm
            + 1  # Output_Tokens_norm
            + len(self.completion_status_categories)
            + len(self.error_type_categories)
        )


def _fit_categories_dropping_zero_variance(df: pd.DataFrame, col: str, label: str) -> list:
    """
    Fit one-hot categories for `col`, but return [] (drop the feature
    entirely) if it has <=1 distinct value in the fitting data -- a
    constant one-hot column is always 1, carries zero information the model
    can use to distinguish samples, and at small train-set sizes every dead
    dimension is pure overfitting surface. Auto-detected per fit, not
    hardcoded to a specific column name, so it stops applying automatically
    once real diversity shows up in future data.
    """
    if col not in df:
        return []
    categories = sorted(df[col].dropna().unique().tolist())
    if len(categories) <= 1:
        if len(categories) == 1:
            print(f"[feature_builder] '{label}' is constant ({categories[0]!r}) in the train split -- "
                  f"dropped (0 dims) instead of encoded as a dead-weight one-hot column.")
        return []
    return categories


def _fit_categories_dropping_high_cardinality(
    df: pd.DataFrame, col: str, label: str, max_cardinality_ratio: float = 0.3
) -> list:
    """
    Like _fit_categories_dropping_zero_variance but guards the OPPOSITE
    failure mode: a categorical column whose distinct-value count is a large
    fraction of the row count is effectively free text (e.g. observed:
    reasoning_type had 87 unique values across 109 rows). One-hot encoding
    that gives the model a near-unique feature per training example -- pure
    memorization surface, not signal. Dropped (0 dims) if
    n_unique/n_rows > max_cardinality_ratio. General mechanism, not
    hardcoded to reasoning_type specifically, so it applies to any future
    column with the same failure mode.
    """
    if col not in df or len(df) == 0:
        return []
    n_unique = df[col].dropna().nunique()
    ratio = n_unique / len(df)
    if ratio > max_cardinality_ratio:
        print(f"[feature_builder] '{label}' has {n_unique} unique values across {len(df)} rows "
              f"(ratio={ratio:.2f} > {max_cardinality_ratio}) -- too high-cardinality/free-text to "
              f"one-hot usefully, dropped (0 dims). The query_embedding already captures this kind "
              f"of semantic signal without the sparse tabular encoding.")
        return []
    return _fit_categories_dropping_zero_variance(df, col, label)


# Fixed, deterministic domain taxonomy -- NOT fit from data. Macro categories
# are a designed grouping (substring rules below), so every train/val/test
# split gets the same dimensions regardless of which raw `domain` strings
# happen to appear in that particular split. This is deliberately different
# from the fit-from-data approach used for other categoricals.
_DOMAIN_MACRO_RULES = [
    ("Combinatorics_Discrete", ["combinatorics", "probability", "discrete", "graph"]),
    ("Number_Theory", ["number theory"]),
    ("Algebra", ["algebra", "equation"]),
    ("Geometry", ["geometry"]),
    ("Olympiad_Logic", ["olympiad", "logic", "game"]),
]
DOMAIN_MACRO_CATEGORIES = [name for name, _ in _DOMAIN_MACRO_RULES] + ["Other"]


def categorize_domain_macro(raw_domain) -> str:
    d = str(raw_domain).lower()
    for name, keywords in _DOMAIN_MACRO_RULES:
        if any(kw in d for kw in keywords):
            return name
    return "Other"


def fit_encoders(
    df: pd.DataFrame,
    difficulty_ordinal_order: Optional[Sequence[str]] = None,
    reasoning_depth_ordinal_order: Optional[Sequence[str]] = None,
    numeric_detection_threshold: float = 0.95,
    reasoning_type_max_cardinality_ratio: float = 0.3,
    model_col: str = "model",
    completion_status_col: str = "Completion_Status",
) -> FeatureEncoders:
    """
    numeric_detection_threshold: fraction of non-null values that must be
    numeric-castable for a column to be treated as z-scored numeric instead
    of one-hot categorical. As of the mapped benchmark, difficulty and
    reasoning_depth are unified integer scales (difficulty: 4-10,
    reasoning_depth: 3-6) and will hit this path automatically -- no
    ordinal_order needed anymore. ordinal_order params are kept for the case
    where a future benchmark version reverts to unmapped text labels.
    """
    enc = FeatureEncoders()

    # domain: prefer precomputed dummy columns (domain_Algebra, domain_Other, ...)
    # if the benchmark file already provides them -- this is now the expected
    # path since domain bucketing was moved upstream into the CSV itself.
    # Falls back to bucketing a raw `domain` text column for older benchmark
    # files that haven't been through that preprocessing yet.
    expected_domain_cols = [f"domain_{name}" for name in DOMAIN_MACRO_CATEGORIES]
    if all(c in df.columns for c in expected_domain_cols):
        enc.domain_precomputed_columns = expected_domain_cols
        enc.domain_categories = []
    else:
        missing = [c for c in expected_domain_cols if c not in df.columns]
        if "domain" in df.columns:
            print(f"[feature_builder] precomputed domain_* columns not fully present "
                  f"(missing {missing}) -- falling back to bucketing raw `domain` text column.")
        enc.domain_categories = list(DOMAIN_MACRO_CATEGORIES)
    enc.reasoning_type_categories = _fit_categories_dropping_high_cardinality(
        df, "reasoning_type", "reasoning_type", reasoning_type_max_cardinality_ratio
    )
    enc.answer_type_categories = _fit_categories_dropping_zero_variance(df, "Answer_Type", "Answer_Type")

    if difficulty_ordinal_order:
        enc.difficulty_ordinal_order = list(difficulty_ordinal_order)
    elif "difficulty" in df:
        numeric_vals = pd.to_numeric(df["difficulty"], errors="coerce")
        frac_numeric = numeric_vals.notna().mean() if len(df) else 0.0
        if frac_numeric >= numeric_detection_threshold:
            enc.difficulty_is_numeric = True
            enc.difficulty_mu_sigma = _zscore_fit(numeric_vals.values)
        else:
            enc.difficulty_categories = _fit_categories_dropping_zero_variance(
                df.assign(difficulty=df["difficulty"].astype(str)), "difficulty", "difficulty"
            )

    if reasoning_depth_ordinal_order:
        enc.reasoning_depth_ordinal_order = list(reasoning_depth_ordinal_order)
    elif "reasoning_depth" in df:
        numeric_vals = pd.to_numeric(df["reasoning_depth"], errors="coerce")
        frac_numeric = numeric_vals.notna().mean() if len(df) else 0.0
        if frac_numeric >= numeric_detection_threshold:
            enc.reasoning_depth_is_numeric = True
            enc.reasoning_depth_mu_sigma = _zscore_fit(numeric_vals.values)
        else:
            enc.reasoning_depth_categories = _fit_categories_dropping_zero_variance(
                df.assign(reasoning_depth=df["reasoning_depth"].astype(str)), "reasoning_depth", "reasoning_depth"
            )

    if "solution_steps" in df:
        enc.solution_steps_mu_sigma = _zscore_fit(pd.to_numeric(df["solution_steps"], errors="coerce").values)

    # log1p first -- raw character length is right-skewed (a few very long
    # queries would otherwise dominate the z-score scale); log1p compresses
    # that tail before normalizing, per the reviewed proposal.
    q_len = df["query"].astype(str).str.len().values.astype(float) if "query" in df else np.array([0.0])
    enc.question_length_mu_sigma = _zscore_fit(np.log1p(q_len))

    if completion_status_col in df:
        enc.completion_status_categories = sorted(df[completion_status_col].dropna().unique().tolist())
    if "Error_Type" in df:
        enc.error_type_categories = sorted(df["Error_Type"].dropna().unique().tolist())

    for col, attr in [
        ("Cost", "cost_mu_sigma"),
        ("Latency", "latency_mu_sigma"),
        ("Input_Tokens", "input_tokens_mu_sigma"),
        ("Output_Tokens", "output_tokens_mu_sigma"),
    ]:
        if col in df:
            setattr(enc, attr, _zscore_fit(pd.to_numeric(df[col], errors="coerce").values))

    # Completion_Reliability per model = fraction of "completed" status.
    # Assumption: the "success" label in Completion_Status is literally the
    # string "completed" (case-insensitive). Verify against real category
    # set before trusting this in production.
    if completion_status_col in df and model_col in df:
        status_norm = df[completion_status_col].astype(str).str.lower()
        is_complete = status_norm.eq("completed")
        reliability = (
            pd.DataFrame({model_col: df[model_col], "_ok": is_complete})
            .groupby(model_col)["_ok"]
            .mean()
        )
        enc.completion_reliability_by_model = reliability.to_dict()

    return enc


# --------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------

def _onehot(series: pd.Series, categories: list) -> np.ndarray:
    if not categories:
        return np.zeros((len(series), 0), dtype=np.float32)
    idx = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(series), len(categories)), dtype=np.float32)
    for row_i, v in enumerate(series.values):
        j = idx.get(v)
        if j is not None:
            out[row_i, j] = 1.0
    return out


def build_node_metadata(df: pd.DataFrame, enc: FeatureEncoders) -> np.ndarray:
    """
    Phase 2/3: per-row metadata vector (row granularity = router_data row,
    caller is responsible for subsetting to unique query rows, exactly as
    prepare_data_for_GNN already does for query_embedding/task_embedding).

    NOTE: domain macro-category is deliberately NOT part of this vector --
    it's the Reasoning node's own feature now (see
    build_reasoning_node_features / assign_reasoning_node_index), not
    something folded into the Question node. Everything else here still
    concatenates into the Question node's feature vector as before.
    """
    parts = []

    parts.append(_onehot(df.get("reasoning_type", pd.Series([None] * len(df))), enc.reasoning_type_categories))

    if enc.difficulty_ordinal_order:
        order = {c: i for i, c in enumerate(enc.difficulty_ordinal_order)}
        diff_vals = df.get("difficulty", pd.Series([None] * len(df))).map(order).astype(float)
        diff_vals = diff_vals.fillna(np.nanmean(list(order.values())) if order else 0.0)
        parts.append(diff_vals.values.reshape(-1, 1).astype(np.float32))
    elif enc.difficulty_is_numeric:
        diff_vals = pd.to_numeric(df.get("difficulty", pd.Series([np.nan] * len(df))), errors="coerce").values
        parts.append(_zscore_apply(diff_vals, *enc.difficulty_mu_sigma).reshape(-1, 1).astype(np.float32))
    else:
        diff_series = df.get("difficulty", pd.Series([None] * len(df))).apply(lambda v: None if pd.isna(v) else str(v))
        parts.append(_onehot(diff_series, enc.difficulty_categories))

    if enc.reasoning_depth_ordinal_order:
        order = {c: i for i, c in enumerate(enc.reasoning_depth_ordinal_order)}
        rd_vals = df.get("reasoning_depth", pd.Series([None] * len(df))).map(order).astype(float)
        rd_vals = rd_vals.fillna(np.nanmean(list(order.values())) if order else 0.0)
        parts.append(rd_vals.values.reshape(-1, 1).astype(np.float32))
    elif enc.reasoning_depth_is_numeric:
        rd_vals = pd.to_numeric(df.get("reasoning_depth", pd.Series([np.nan] * len(df))), errors="coerce").values
        parts.append(_zscore_apply(rd_vals, *enc.reasoning_depth_mu_sigma).reshape(-1, 1).astype(np.float32))
    else:
        rd_series = df.get("reasoning_depth", pd.Series([None] * len(df))).apply(lambda v: None if pd.isna(v) else str(v))
        parts.append(_onehot(rd_series, enc.reasoning_depth_categories))

    ss = pd.to_numeric(df.get("solution_steps", pd.Series([np.nan] * len(df))), errors="coerce").values
    parts.append(_zscore_apply(ss, *enc.solution_steps_mu_sigma).reshape(-1, 1).astype(np.float32))

    parts.append(_onehot(df.get("Answer_Type", pd.Series([None] * len(df))), enc.answer_type_categories))

    q_len = df.get("query", pd.Series([""] * len(df))).astype(str).str.len().values.astype(float)
    parts.append(_zscore_apply(np.log1p(q_len), *enc.question_length_mu_sigma).reshape(-1, 1).astype(np.float32))

    return np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=np.float32)


def build_reasoning_node_features(enc: FeatureEncoders) -> np.ndarray:
    """
    Feature matrix for the shared Reasoning prototype nodes: one row per
    domain macro-category, in the same fixed order as enc.reasoning_categories.
    Each prototype node's feature is simply its own one-hot identity within
    that fixed set (an identity matrix) -- there is one node per category,
    shared across every query in that bucket, not one node per query.
    """
    n = enc.reasoning_node_dim
    return np.eye(n, dtype=np.float32) if n > 0 else np.zeros((0, 0), dtype=np.float32)


def assign_reasoning_node_index(df: pd.DataFrame, enc: FeatureEncoders) -> np.ndarray:
    """
    Per-row index (0-based) into the Reasoning prototype node set
    (enc.reasoning_categories) that each router_data row's query belongs to.
    Caller subsets to unique query rows the same way it does for
    query_embedding/task_embedding/node_metadata.
    """
    categories = enc.reasoning_categories
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    other_idx = cat_to_idx.get("Other", 0)

    if enc.domain_precomputed_columns:
        domain_block = df.reindex(columns=enc.domain_precomputed_columns, fill_value=0).fillna(0)
        return domain_block.values.argmax(axis=1).astype(np.int64)

    domain_macro_series = df.get("domain", pd.Series([None] * len(df))).apply(categorize_domain_macro)
    return domain_macro_series.map(lambda c: cat_to_idx.get(c, other_idx)).values.astype(np.int64)


def build_edge_features(df: pd.DataFrame, enc: FeatureEncoders) -> np.ndarray:
    """Phase 4: per-edge (per router_data row) feature vector."""
    correct = pd.to_numeric(df.get("Correct", pd.Series([0] * len(df))), errors="coerce").fillna(0).values.reshape(-1, 1).astype(np.float32)

    cost = pd.to_numeric(df.get("Cost", pd.Series([np.nan] * len(df))), errors="coerce").values
    cost_n = _zscore_apply(cost, *enc.cost_mu_sigma).reshape(-1, 1).astype(np.float32)

    latency = pd.to_numeric(df.get("Latency", pd.Series([np.nan] * len(df))), errors="coerce").values
    latency_n = _zscore_apply(latency, *enc.latency_mu_sigma).reshape(-1, 1).astype(np.float32)

    in_tok = pd.to_numeric(df.get("Input_Tokens", pd.Series([np.nan] * len(df))), errors="coerce").values
    in_tok_n = _zscore_apply(in_tok, *enc.input_tokens_mu_sigma).reshape(-1, 1).astype(np.float32)

    out_tok = pd.to_numeric(df.get("Output_Tokens", pd.Series([np.nan] * len(df))), errors="coerce").values
    out_tok_n = _zscore_apply(out_tok, *enc.output_tokens_mu_sigma).reshape(-1, 1).astype(np.float32)

    status_oh = _onehot(df.get("Completion_Status", pd.Series([None] * len(df))), enc.completion_status_categories)
    error_oh = _onehot(df.get("Error_Type", pd.Series([None] * len(df))), enc.error_type_categories)

    return np.concatenate(
        [correct, cost_n, latency_n, in_tok_n, out_tok_n, status_oh, error_oh], axis=1
    )


def find_degenerate_queries(df: pd.DataFrame, group_col: str, correct_col: str = "Correct") -> set:
    """
    Returns the set of group_col values (query_ids) where every row has the
    same Correct value (all-0 or all-1) -- i.e. no model disagreement, so the
    utility label for that query is driven entirely by cost/latency/
    reliability, not by any correctness/skill signal. Not necessarily noise
    (the ranking is still well-defined), but a distinct, weaker kind of
    signal than the 92.9%-of-cases where models disagree. Exposed so the
    caller can optionally exclude these from training to concentrate on the
    discriminative examples -- opt-in, not applied by default.
    """
    correctness_spread = df.groupby(group_col)[correct_col].nunique()
    return set(correctness_spread[correctness_spread <= 1].index)


def compute_utility(
    df: pd.DataFrame,
    enc: FeatureEncoders,
    w_success: float = 1.0,
    w_cost: float = 0.3,
    w_latency: float = 0.3,
    w_output_tokens: float = 0.2,
    w_completion_reliability: float = 0.5,
    model_col: str = "model",
) -> np.ndarray:
    """
    Phase 6 revised utility function (usable-today subset):
        U = w1*Correct - w2*Cost_norm - w3*Latency_norm
            - w4*Output_Tokens_norm + w5*Completion_Reliability

    NOTE: weights are placeholders (not tuned against the paper's protocol
    or a validation sweep). Treat as a hyperparameter to search once real
    data is available -- see "Next actions".
    """

    correct = pd.to_numeric(df.get("Correct", pd.Series([0] * len(df))), errors="coerce").fillna(0).values

    cost = pd.to_numeric(df.get("Cost", pd.Series([np.nan] * len(df))), errors="coerce").values
    cost_n = _zscore_apply(cost, *enc.cost_mu_sigma)

    latency = pd.to_numeric(df.get("Latency", pd.Series([np.nan] * len(df))), errors="coerce").values
    latency_n = _zscore_apply(latency, *enc.latency_mu_sigma)

    out_tok = pd.to_numeric(df.get("Output_Tokens", pd.Series([np.nan] * len(df))), errors="coerce").values
    out_tok_n = _zscore_apply(out_tok, *enc.output_tokens_mu_sigma)

    reliability = df.get(model_col, pd.Series(["__unknown__"] * len(df))).map(
        lambda m: enc.completion_reliability_by_model.get(m, 0.5)
    ).values.astype(float)


    utility = (
        w_success * correct
        - w_cost * cost_n
        - w_latency * latency_n
        - w_output_tokens * out_tok_n
        + w_completion_reliability * reliability
    )
    return utility.astype(np.float32)