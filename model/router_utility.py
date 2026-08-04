"""
Utility-score and ranking-metric calculations for the graph router.

Extracted from multi_task_graph_router.py -- no logic changes, only
relocation. These are pure functions (no dependency on
graph_router_prediction's internal state) so they can be unit-tested
and reused independently of the training/inference pipeline.
"""

import numpy as np
import pandas as pd


# Maps each utility_weights key to the raw router_data column(s) whose
# variance determines whether that weight can actually influence a ranking.
# w_completion_reliability -> Completion_Status because reliability is
# derived directly from it (constant status => constant reliability).
_UTILITY_WEIGHT_SOURCE_COLUMNS = {
    'w_success': ['Correct'],
    'w_cost': ['Cost'],
    'w_latency': ['Latency'],
    'w_output_tokens': ['Output_Tokens'],
    'w_completion_reliability': ['Completion_Status'],
}


def calculate_query_utility(
    query_id: str | int,
    df: pd.DataFrame,
    weights: dict = None,
    query_id_col: str = 'query_id',
) -> pd.DataFrame:
    """
    Calculates utility scores across all models for a specific query ID using
    exact dataset column names.

    Parameters:
    -----------
    query_id : str or int
        Unique identifier for the target query.
    df : pd.DataFrame
        DataFrame containing query benchmarking records across candidate models.
    weights : dict, optional
        Utility weights matching config.yaml specifications.
    query_id_col : str, optional
        Column to filter on. Defaults to 'query_id'; pass 'row_id' when
        calling against router_data.csv-derived DataFrames in this codebase,
        which use 'row_id' as the per-query identifier.

    Returns:
    --------
    pd.DataFrame
        Table of candidate models ranked by calculated utility score.
    """
    # Default weights matching config.yaml
    if weights is None:
        weights = {
            'w_success': 1.0,
            'w_cost': 0.3,
            'w_latency': 0.3,
            'w_output_tokens': 0.2,
            'w_completion_reliability': 0.5
        }

    # Filter dataframe for the given query
    query_df = df[df[query_id_col] == query_id].copy()
    if query_df.empty:
        raise ValueError(f"Query ID '{query_id}' not found in the dataset.")

    # Extract key metric series using exact column names
    success = query_df['Correct'].astype(float)
    cost = query_df['Cost'].astype(float)
    latency = query_df['Latency'].astype(float)
    out_tokens = query_df['Output_Tokens'].astype(float)

    # Reliability: 1.0 if status indicates success/completion, otherwise 0.0
    reliability = query_df['Completion_Status'].apply(
        lambda x: 1.0 if x in [1, True, 'Success', 'SUCCESS', 'Completed', 'completed'] else (float(x) if str(x).replace('.', '', 1).isdigit() else 0.0)
    )

    # Per-query min-max normalization function for penalty metrics
    def min_max_norm(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else pd.Series(0.0, index=series.index)

    # Calculate normalized penalties relative to candidate models for this query
    cost_norm = min_max_norm(cost)
    latency_norm = min_max_norm(latency)
    out_tokens_norm = min_max_norm(out_tokens)

    # Utility score calculation
    query_df['utility_score'] = (
        weights['w_success'] * success
        + weights['w_completion_reliability'] * reliability
        - weights['w_cost'] * cost_norm
        - weights['w_latency'] * latency_norm
        - weights['w_output_tokens'] * out_tokens_norm
    )

    # Select key summary columns and sort by highest utility score
    display_cols = [
        'model', 'utility_score', 'Correct', 'Cost',
        'Latency', 'Output_Tokens', 'Completion_Status'
    ]

    result = (
        query_df[display_cols]
        .sort_values(by='utility_score', ascending=False)
        .reset_index(drop=True)
    )

    return result


def compute_ranking_metrics(predicted_scores: dict, ground_truth_df: pd.DataFrame) -> dict:
    """
    Compare the router's predicted per-model scores against the
    calculate_query_utility() ground-truth ranking for one query.

    predicted_scores: {model_name: predicted_score}
    ground_truth_df: output of calculate_query_utility() -- already sorted
        best-to-worst by 'utility_score'.
    """

    print(predicted_scores)
    gt = ground_truth_df.copy()
    gt['predicted_score'] = gt['model'].map(predicted_scores)
    if gt['predicted_score'].isna().any():
        missing = gt.loc[gt['predicted_score'].isna(), 'model'].tolist()
        raise ValueError(f"compute_ranking_metrics: no predicted score for model(s) {missing} -- "
                          f"predicted_scores keys must match ground_truth_df['model'] exactly.")

    n = len(gt)
    gt_rank = gt['utility_score'].rank(ascending=False, method='first')
    pred_rank = gt['predicted_score'].rank(ascending=False, method='first')

    # Spearman rank correlation (no scipy dependency: Pearson correlation of ranks).
    if n > 1 and gt_rank.std() > 0 and pred_rank.std() > 0:
        spearman = float(np.corrcoef(gt_rank, pred_rank)[0, 1])
    else:
        spearman = float('nan')

    top1_match = bool(gt.loc[gt_rank.idxmin(), 'model'] == gt.loc[pred_rank.idxmin(), 'model'])

    # NDCG@3 -- ranks candidates by predicted score, scores against ground-truth utility.
    k = min(3, n)
    pred_order = gt.sort_values('predicted_score', ascending=False)
    rel = pred_order['utility_score'].values[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel * discounts))
    ideal_rel = gt.sort_values('utility_score', ascending=False)['utility_score'].values[:k]
    idcg = float(np.sum(ideal_rel * discounts))
    ndcg_at_3 = dcg / idcg if idcg > 1e-12 else 0.0

    mean_abs_rank_error = float(np.mean(np.abs(gt_rank.values - pred_rank.values)))

    return {
        'top1_match': top1_match,
        'spearman_rank_corr': spearman,
        'ndcg_at_3': ndcg_at_3,
        'mean_abs_rank_error': mean_abs_rank_error,
        'n_models': n,
    }


def auto_drop_zero_variance_utility_weights(data_df: pd.DataFrame, weights: dict) -> dict:
    """
    Zero out (and warn about) any utility_weights entry whose source column
    is constant across data_df.

    A weight on a constant column contributes an identical additive term to
    every (query, model) edge's utility -- it cancels out under the
    per-query softmax and never changes which model ranks highest, so it's
    dead weight that misleadingly looks "active" in config. This mirrors the
    zero-variance auto-drop already applied to node metadata features, but
    for the utility formula itself, and logs what happened instead of
    silently doing nothing.

    Non-destructive: returns a new dict, doesn't mutate the input weights.
    """
    resolved = dict(weights)
    for weight_key, source_cols in _UTILITY_WEIGHT_SOURCE_COLUMNS.items():
        if weight_key not in resolved or resolved[weight_key] == 0:
            continue
        for col in source_cols:
            if col not in data_df.columns:
                continue
            if data_df[col].nunique(dropna=False) <= 1:
                print(f"[utility_weights][WARN] '{col}' is constant "
                      f"(value={data_df[col].iloc[0]!r}) across all rows -- "
                      f"'{weight_key}'={resolved[weight_key]} contributes a "
                      f"constant offset that cannot affect ranking. "
                      f"Zeroing '{weight_key}' for this run; it will "
                      f"automatically re-activate once '{col}' actually varies.")
                resolved[weight_key] = 0.0
    return resolved


def resolve_utility_weights(config: dict) -> dict:
    """
    Resolve the active utility_weights dict from config.yaml.

    Supports the new named-profile format:
        utility_scenario: "cost_first"
        utility_weight_profiles:
          cost_first: {w_success: ..., w_cost: ..., ...}
          balance: {...}
          performance_first: {...}
    and falls back to a flat `utility_weights:` dict (old format) or the
    hardcoded default if neither is present, so existing configs and
    direct-call sites don't break.
    """
    profiles = config.get('utility_weight_profiles')
    if profiles:
        scenario = config.get('utility_scenario')
        if scenario is None:
            raise ValueError(
                "config has 'utility_weight_profiles' but no 'utility_scenario' "
                "selecting which one is active. Set utility_scenario to one of: "
                f"{list(profiles)}"
            )
        if scenario not in profiles:
            raise ValueError(
                f"utility_scenario '{scenario}' not found in utility_weight_profiles "
                f"(available: {list(profiles)})"
            )
        return dict(profiles[scenario])

    # Old flat format / direct fallback.
    return config.get('utility_weights', {
        'w_success': 1.0, 'w_cost': 0.3, 'w_latency': 0.3,
        'w_output_tokens': 0.2, 'w_completion_reliability': 0.5,
    })
