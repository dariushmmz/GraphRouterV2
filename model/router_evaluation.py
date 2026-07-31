"""
Router evaluation: Relative Utility Gap Closed + supporting metrics.

Why this exists (replaces plain Top-1 accuracy as the headline number):
Top-1 accuracy penalizes the router equally whether it picked the 2nd-best
model at 1% worse utility or the worst model at 90% worse utility, and it
gives no credit for cases where several models are essentially tied. On a
~71-query dataset a single query flipping Top-1 shifts accuracy by ~1.4pp,
so it's also a noisy number on its own -- see run_kfold_cv's docstring in
graph_nn.py for the same variance concern driving the CV protocol below.

Everything here operates on a long-format DataFrame with one row per
(query, model) edge -- i.e. exactly the shape data_df already has in
multi_task_graph_router.py (num_llms consecutive rows per query).
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Core evaluator
# ----------------------------------------------------------------------------
def evaluate_router_performance(
    df: pd.DataFrame,
    pred_col: str = "predicted_utility",
    gold_col: str = "utility",
    query_id_col: str = "query_id",
    model_col: str = "model",
    cost_col: str = "Cost",
    correctness_col: str = "Correct",
    latency_col: str = "Latency",
    top_k_tolerance: float = 0.05,
    ndcg_k: int = 3,
    seed: int = 42,
) -> dict:
    """
    Evaluate router predictions against Oracle, Single-Best-Model (SBM),
    Random, and Cost-Optimal baselines.

    Expects `df` grouped so that every query contributes exactly the same
    set of `model_col` values (one row per query-model edge), with:
      - pred_col: the router's predicted utility/score for that edge
      - gold_col: the ground-truth utility for that edge
    cost_col / correctness_col / latency_col are optional; metrics that
    need a missing column are skipped (reported as None) rather than
    raising, so this works even before all columns are wired up.
    """
    required = {query_id_col, model_col, gold_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"evaluate_router_performance: missing required column(s): {missing}")

    has_cost = cost_col in df.columns
    has_correct = correctness_col in df.columns
    has_latency = latency_col is not None and latency_col in df.columns

    # ---- Per-query selections ----
    oracle_idx = df.groupby(query_id_col)[gold_col].idxmax()
    oracle_df = df.loc[oracle_idx]

    sbm_model = df.groupby(model_col)[gold_col].mean().idxmax()
    sbm_df = df[df[model_col] == sbm_model]
    # Reindex SBM to one row per query (in case SBM appears more than once,
    # which shouldn't happen for a clean rectangular block but is cheap to guard).
    sbm_df = sbm_df.drop_duplicates(subset=[query_id_col])

    router_idx = df.groupby(query_id_col)[pred_col].idxmax()
    router_df = df.loc[router_idx]

    rng = np.random.RandomState(seed)
    random_df = (
        df.groupby(query_id_col, group_keys=False)
        .apply(lambda x: x.sample(1, random_state=seed))
        .reset_index(drop=True)
    )

    cost_optimal_df = None
    if has_cost:
        cost_idx = df.groupby(query_id_col)[cost_col].idxmin()
        cost_optimal_df = df.loc[cost_idx]

    # ---- Mean utilities ----
    u_oracle = oracle_df[gold_col].mean()
    u_sbm = sbm_df[gold_col].mean()
    u_router = router_df[gold_col].mean()
    u_random = random_df[gold_col].mean()
    u_cost_optimal = cost_optimal_df[gold_col].mean() if cost_optimal_df is not None else None

    # ---- A. Relative Utility Gap Closed (primary metric) ----
    # Reported against two different baselines because they represent two
    # different deployment intents:
    #   - vs. SBM (accuracy-first): "would you have done better just always
    #     calling the single most-correct model?" Appropriate when the
    #     router is meant to compete with SBM on quality.
    #   - vs. Cost-Optimal: "would you have done better just always calling
    #     the cheapest model?" Appropriate when the router's utility_weights
    #     are deliberately cost/latency-leaning, in which case SBM alone is
    #     the wrong yardstick and can make a working cost-optimized router
    #     look like a failure when it's actually beating the cheap baseline.
    denom = u_oracle - u_sbm
    gap_closed = ((u_router - u_sbm) / denom * 100) if abs(denom) > 1e-12 else 0.0

    gap_closed_cost_optimal = None
    if u_cost_optimal is not None:
        denom_cost = u_oracle - u_cost_optimal
        gap_closed_cost_optimal = (
            (u_router - u_cost_optimal) / denom_cost * 100
        ) if abs(denom_cost) > 1e-12 else 0.0

    # ---- B. Selection win-rate ----
    top1_match_pct = (router_df[model_col].values == oracle_df[model_col].values).mean() * 100

    max_per_query = df.groupby(query_id_col)[gold_col].transform("max")
    within_tol = router_df[gold_col].values >= (
        max_per_query.loc[router_df.index].values * (1 - top_k_tolerance)
    )
    top_k_match_pct = within_tol.mean() * 100

    # ---- C. Pareto: accuracy / cost / latency vs SBM ----
    accuracy_retention_pct = cost_reduction_pct = latency_reduction_pct = None
    router_acc = sbm_acc = None
    if has_correct:
        router_acc = router_df[correctness_col].mean() * 100
        sbm_acc = sbm_df[correctness_col].mean() * 100
        if sbm_acc > 1e-12:
            accuracy_retention_pct = (router_acc / sbm_acc) * 100
    if has_cost:
        sbm_cost = sbm_df[cost_col].sum()
        router_cost = router_df[cost_col].sum()
        if sbm_cost > 1e-12:
            cost_reduction_pct = ((sbm_cost - router_cost) / sbm_cost) * 100
    if has_latency:
        sbm_latency = sbm_df[latency_col].sum()
        router_latency = router_df[latency_col].sum()
        if sbm_latency > 1e-12:
            latency_reduction_pct = ((sbm_latency - router_latency) / sbm_latency) * 100

    # ---- D. NDCG@K ----
    ndcg = _mean_ndcg_at_k(df, pred_col=pred_col, gold_col=gold_col,
                            query_id_col=query_id_col, k=ndcg_k)

    results = {
        "u_oracle": u_oracle,
        "u_sbm": u_sbm,
        "u_router": u_router,
        "u_random": u_random,
        "u_cost_optimal": u_cost_optimal,
        "sbm_model": sbm_model,
        "gap_closed_pct": gap_closed,
        "gap_closed_vs_cost_optimal_pct": gap_closed_cost_optimal,
        "top1_match_pct": top1_match_pct,
        f"top_k_match_pct_tol{top_k_tolerance:g}": top_k_match_pct,
        "router_accuracy_pct": router_acc,
        "sbm_accuracy_pct": sbm_acc,
        "accuracy_retention_pct": accuracy_retention_pct,
        "cost_reduction_pct": cost_reduction_pct,
        "latency_reduction_pct": latency_reduction_pct,
        f"ndcg_at_{ndcg_k}": ndcg,
        "n_queries": df[query_id_col].nunique(),
    }

    _print_report(results, top_k_tolerance, ndcg_k)
    return results


def _mean_ndcg_at_k(df, pred_col, gold_col, query_id_col, k):
    """
    Mean NDCG@k across queries, ranking candidate models by predicted
    utility and scoring against ground-truth utility as graded relevance.
    Verifies the router's *ranking* quality beyond just the Top-1 pick --
    a router that consistently puts the true-best model in position 2 gets
    partial, not zero, credit here (unlike Top-1 accuracy).
    """
    scores = []
    for _, g in df.groupby(query_id_col):
        g_sorted_pred = g.sort_values(pred_col, ascending=False)
        rel = g_sorted_pred[gold_col].values[:k]
        discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
        dcg = np.sum(rel * discounts)

        ideal_rel = g.sort_values(gold_col, ascending=False)[gold_col].values[:k]
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_rel) + 2))
        idcg = np.sum(ideal_rel * ideal_discounts)

        scores.append(dcg / idcg if idcg > 1e-12 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def _print_report(r, top_k_tolerance, ndcg_k):
    print("=== Router Evaluation Report ===")
    print(f"Single Best Model (SBM)     : {r['sbm_model']}")
    print(f"Queries evaluated           : {r['n_queries']}")
    print(f"Mean Utility - Oracle       : {r['u_oracle']:.4f}")
    print(f"Mean Utility - Router       : {r['u_router']:.4f}")
    print(f"Mean Utility - SBM          : {r['u_sbm']:.4f}")
    print(f"Mean Utility - Random       : {r['u_random']:.4f}")
    if r["u_cost_optimal"] is not None:
        print(f"Mean Utility - Cost-Optimal : {r['u_cost_optimal']:.4f}")
    print("-" * 40)
    print(f"Utility Gap Closed (vs SBM) : {r['gap_closed_pct']:.2f}%  "
          f"(100% = matches Oracle, 0% = no better than SBM, <0% = worse than SBM)")
    if r["gap_closed_vs_cost_optimal_pct"] is not None:
        print(f"Utility Gap Closed (vs Cost-Optimal): {r['gap_closed_vs_cost_optimal_pct']:.2f}%  "
              f"(100% = matches Oracle, 0% = no better than always-cheapest, <0% = worse than always-cheapest)")
    print(f"Top-1 Oracle Match Rate     : {r['top1_match_pct']:.2f}%")
    print(f"Top-K Match (within {top_k_tolerance:.0%})  : "
          f"{r[f'top_k_match_pct_tol{top_k_tolerance:g}']:.2f}%")
    print(f"NDCG@{ndcg_k}                    : {r[f'ndcg_at_{ndcg_k}']:.4f}")
    if r["accuracy_retention_pct"] is not None:
        print(f"Accuracy vs SBM             : {r['router_accuracy_pct']:.1f}% "
              f"(SBM: {r['sbm_accuracy_pct']:.1f}%, retention: {r['accuracy_retention_pct']:.1f}%)")
    if r["cost_reduction_pct"] is not None:
        print(f"Cost Reduction vs SBM       : {r['cost_reduction_pct']:.2f}%")
    if r["latency_reduction_pct"] is not None:
        print(f"Latency Reduction vs SBM    : {r['latency_reduction_pct']:.2f}%")


# ----------------------------------------------------------------------------
# Hook for run_kfold_cv (graph_nn.py): aggregate OUT-OF-FOLD predictions
# ----------------------------------------------------------------------------
def build_oof_predictions_df(data_df, num_llms, oof_predicted_utility, gold_col="utility",
                              query_id_col=None, cost_col=None, correctness_col=None,
                              latency_col=None):
    """
    Assemble the long-format DataFrame evaluate_router_performance() expects,
    from the out-of-fold predictions produced across all K folds of
    run_kfold_cv.

    This is the piece the previous per-fold summary (mean/std of each fold's
    own held-out accuracy) was missing: averaging 5 independent small-sample
    accuracies is noisier than pooling all ~71 queries' out-of-fold
    predictions into one table and scoring once. Every query in `data_df`
    gets scored by exactly the model that had it held out, so this is a
    clean, non-leaky, low-variance estimate.

    Args:
        data_df: the original rectangular router_data DataFrame (num_llms
            consecutive rows per query, same row order used throughout
            multi_task_graph_router.py).
        num_llms: models per query (== config['llm_num']).
        oof_predicted_utility: 1D array/tensor aligned row-for-row with
            data_df, containing each row's out-of-fold predicted utility
            (collect this in run_kfold_cv by writing each fold's holdout
            predictions into a full-length buffer at the holdout row
            indices, instead of only recording the fold's mean).
        query_id_col: existing column to use as query id; if None, a
            positional block id (row_index // num_llms) is generated --
            matches how edge_org_id is built elsewhere in this codebase.
    """
    out = data_df.copy().reset_index(drop=True)
    out["predicted_utility"] = np.asarray(oof_predicted_utility).reshape(-1)

    if query_id_col is None or query_id_col not in out.columns:
        out["query_id"] = out.index // num_llms
        query_id_col = "query_id"

    rename = {}
    if gold_col not in out.columns:
        raise ValueError(f"build_oof_predictions_df: gold column '{gold_col}' not found in data_df; "
                          f"pass the column that holds fb.compute_utility(...) output.")

    return out, {
        "query_id_col": query_id_col,
        "gold_col": gold_col,
        "cost_col": cost_col,
        "correctness_col": correctness_col,
        "latency_col": latency_col,
    }