import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import numpy as np
import torch
from graph_nn import form_data, EncoderDecoderNet, run_kfold_cv
from data_processing.utils import ask_and_save_feedback, ensure_2d, parse_embedding_field
from data_processing import feature_builder as fb
import pandas as pd
import json
import pickle
import re
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
print("---------------> ALL IMPORTED")


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


def plot_ground_truth_ranking(ground_truth_df: pd.DataFrame,
                               output_dir: str, query_id, task_id=None, ranking_metrics=None):
    """
    Bar chart of the calculate_query_utility() ground-truth ranking alone --
    candidate models ordered left-to-right best-to-worst by utility_score.
    (Predicted scores are shown separately by make_plot(); this chart is
    ground truth only, not a predicted-vs-ground-truth comparison.)

    NOTE on scale: 'utility_score' here is calculate_query_utility()'s raw
    success/cost/latency/output-tokens/reliability formula -- NOT the
    softmax-smoothed edge_attr distribution the console's "Top-3 Ground
    Truth LLMs" print uses. The raw formula is a signed sum (success +
    reliability - weighted penalty terms), so negative utility_score values
    here are expected for a model that failed the query/was penalized
    heavily, and are not comparable in scale to the softmax probabilities
    printed to console (which are always in [0, 1] and sum to 1 by
    construction). Both are legitimate ground truths for different
    purposes -- see infer_single_query's docstring.
    """
    import plotly.io as pio
    import plotly.graph_objects as go

    gt = ground_truth_df.sort_values('utility_score', ascending=False).reset_index(drop=True)
    models = gt['model'].tolist()
    gt_scores = gt['utility_score'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=gt_scores, name='Ground Truth (utility_score, raw)',
                          marker_color='#ff7f0e',
                          text=[f"{s:.3f}" for s in gt_scores], textposition='outside'))
    fig.update_layout(
        title=f"Ground-Truth Ranking | Query {query_id}"
              f"{' | Task ' + str(task_id) if task_id is not None else ''}",
        xaxis_title="LLM (ordered by ground-truth rank, best first)",
        yaxis_title="Ground-truth utility_score (raw -- can be negative)",
        template="plotly_white",
        showlegend=False,
    )

    # Ranking metrics (Top-1 Match / Spearman / NDCG@3 / Mean Abs Rank Error),
    # same numbers printed to console -- shown as an on-chart annotation so
    # the saved HTML is self-contained.
    metrics_text = _format_ranking_metrics_annotation(ranking_metrics)
    if metrics_text:
        fig.add_annotation(
            xref="paper", yref="paper", x=1.0, y=1.0,
            xanchor="right", yanchor="top",
            text=metrics_text,
            showarrow=False, align="left",
            bordercolor="#888", borderwidth=1, borderpad=6,
            bgcolor="rgba(255,255,255,0.85)",
        )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ground_truth_ranking_query_{query_id}.html")
    try:
        fig.write_html(output_file)
        print(f"Ground-truth ranking plot saved as: {output_file}")
        pio.renderers.default = "browser"
    except Exception:
        pio.renderers.default = "colab"
    fig.show()
    return output_file


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


def _format_utility_weights_subtitle(utility_weights: dict) -> str:
    """
    Turn a utility_weights dict (w_success/w_cost/w_latency/w_output_tokens/
    w_completion_reliability) into a human-readable "Success=1.0, Cost=0.3, ..."
    string for use as a plot subtitle, so the weighting a chart was produced
    under is visible on the chart itself instead of only in config.yaml.
    """
    if not utility_weights:
        return ""
    label_order = [
        ("w_success", "Success"),
        ("w_cost", "Cost"),
        ("w_latency", "Latency"),
        ("w_output_tokens", "Tokens"),
        ("w_completion_reliability", "Reliability"),
    ]
    parts = [f"{label}={utility_weights[key]}" for key, label in label_order if key in utility_weights]
    # Include any weight keys not in the known set rather than silently dropping them.
    parts += [f"{k}={v}" for k, v in utility_weights.items() if k not in dict(label_order)]
    return "Utility weights: " + ", ".join(parts)


def _format_ranking_metrics_annotation(ranking_metrics: dict) -> str:
    """
    Turn a compute_ranking_metrics() dict into the same four-line summary
    printed to console ("Top-1 Match", "Spearman Rank Corr", "NDCG@3",
    "Mean Abs Rank Error"), for use as an on-chart annotation.
    """
    if not ranking_metrics:
        return ""
    return (
        f"Top-1 Match: {ranking_metrics['top1_match']}<br>"
        f"Spearman Rank Corr: {ranking_metrics['spearman_rank_corr']:.4f}<br>"
        f"NDCG@3: {ranking_metrics['ndcg_at_3']:.4f}<br>"
        f"Mean Abs Rank Error: {ranking_metrics['mean_abs_rank_error']:.4f}"
    )


def make_plot(scores, output_dir, scenario, query_id, task_id, utility_weights=None, ranking_metrics=None):
    results_list = []
    for llm, score in scores.items():
        results_list.append({
            "LLM": llm,
            "Score": score,
            "Scenario": scenario
        })

    df_scores = pd.DataFrame(results_list)
    import plotly.io as pio
    import plotly.express as px
    # Compute min/max for y-axis with margin
    y_min = df_scores["Score"].min()
    y_max = df_scores["Score"].max()
    margin = (y_max - y_min) * 0.05  # 5% margin

    # Title reflects the utility_weights the ground truth for this run was
    # computed under (w_success/w_cost/w_latency/w_output_tokens/
    # w_completion_reliability from config.yaml), instead of a fixed,
    # weight-agnostic string.
    weights_subtitle = _format_utility_weights_subtitle(utility_weights)
    title = f"LLM Scores for Query {query_id} | Task {task_id}"
    if weights_subtitle:
        title += f"<br><sup>{weights_subtitle}</sup>"

    # Create interactive grouped bar chart
    fig = px.bar(
        df_scores,
        x="LLM",
        y="Score",
        # color="Scenario",
        barmode="group",
        text=df_scores["Score"].apply(lambda x: f"{x:.3f}"),
        title=title
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="LLM",
        yaxis_title="Score",
        xaxis_tickangle=-45,
        yaxis=dict(showgrid=True, range=[y_min - margin, y_max + margin]),
        # legend_title="Scenario",
        template="plotly_white"
    )

    # Ranking metrics (Top-1 Match / Spearman / NDCG@3 / Mean Abs Rank Error),
    # same numbers printed to console -- shown as an on-chart annotation so
    # the saved HTML is self-contained.
    metrics_text = _format_ranking_metrics_annotation(ranking_metrics)
    if metrics_text:
        fig.add_annotation(
            xref="paper", yref="paper", x=1.0, y=1.0,
            xanchor="right", yanchor="top",
            text=metrics_text,
            showarrow=False, align="left",
            bordercolor="#888", borderwidth=1, borderpad=6,
            bgcolor="rgba(255,255,255,0.85)",
        )

    # Save as HTML file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"llm_scores_query_{query_id}.html")
    try:
        fig.write_html(output_file)
        print(f"Graph saved as: {output_file}")
        pio.renderers.default = "browser"

    except:
        pio.renderers.default = "colab"

    fig.show()


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


class graph_router_prediction:
    def __init__(self, router_data_path, llm_path, llm_embedding_path, config, wandb, query_id=0, task_id=None,
                 adaptive=False, inference=False, benchmark_path=None):
        self.config = config
        self.wandb = wandb
        self.data_df = pd.read_csv(router_data_path)
        if task_id:
            self.data_df = self.data_df[self.data_df['row_id'] == task_id]
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.num_llms = len(self.llm_names)
        self.llm_description_embedding = loadpkl(llm_embedding_path)
        self.llm_dim = self.llm_description_embedding.shape[1]

        # Optional: drop specific LLM(s) entirely instead of dropping every
        # query that's missing them. Useful when incompleteness is
        # SYSTEMATIC to one model (e.g. all 22 incomplete queries missing
        # exactly 'deepseek/deepseek-r1') -- recovers those queries' data for
        # the remaining models rather than discarding them wholesale.
        # Requires BOTH the router_data.model string and the
        # LLM_Descriptions.json key spelled out explicitly -- deliberately
        # NOT auto-matched (see prior discussion: the two use unrelated
        # naming schemes and guessing the correspondence is a separate,
        # unverified concern from data completeness). Example config:
        #   exclude_llms:
        #     - router_id: "deepseek/deepseek-r1"
        #       display_name: "DeepSeek-R1"
        exclude_llms = config.get('exclude_llms', [])
        if exclude_llms:
            router_ids_to_drop = {e['router_id'] for e in exclude_llms}
            display_names_to_drop = {e['display_name'] for e in exclude_llms}
            unknown_display = display_names_to_drop - set(self.llm_names)
            if unknown_display:
                raise ValueError(
                    f"[data_integrity] exclude_llms display_name(s) {unknown_display} not found in "
                    f"LLM_Descriptions.json keys {self.llm_names} -- check spelling."
                )
            self.data_df = self.data_df[~self.data_df['model'].isin(router_ids_to_drop)].reset_index(drop=True)
            keep_idx = [i for i, n in enumerate(self.llm_names) if n not in display_names_to_drop]
            self.llm_description_embedding = self.llm_description_embedding[keep_idx]
            self.llm_names = [self.llm_names[i] for i in keep_idx]
            self.llm_description = {n: self.llm_description[n] for n in self.llm_names}
            self.num_llms = len(self.llm_names)
            # BUGFIX: graph_nn.py's train_validate()/test() read config['llm_num']
            # directly for reshape(-1, config['llm_num']) -- a static YAML value
            # that does NOT auto-update when exclude_llms changes the actual
            # model count. Left stale, this either crashes (shape mismatch) or,
            # worse, silently misaligns predictions to labels if the row counts
            # happen to still divide evenly. Force it in sync here so this can't
            # drift the way edge_dim used to.
            self.config['llm_num'] = self.num_llms
            config['llm_num'] = self.num_llms
            print(f"[data_integrity] excluded LLM(s) router_id={sorted(router_ids_to_drop)} / "
                  f"display_name={sorted(display_names_to_drop)} -- num_llms now {self.num_llms}, "
                  f"llm_names={self.llm_names}")

        # --- BUGFIX: the entire pipeline (edge_org_id/edge_des_id, unique_index_list,
        # train/val/test slicing) assumes every query occupies exactly num_llms
        # CONSECUTIVE rows in llm-canonical order -- pure positional indexing, no
        # query_id lookup anywhere downstream. If len(data_df) isn't divisible by
        # num_llms (as observed: 664 rows / 7 llms), at least one query is missing
        # a model run, and int(len/num_llms) truncation silently desyncs every
        # block boundary after the first incomplete query, not just the reshape
        # at the end. Fix: group by query_id, keep only queries with all
        # num_llms models present exactly once, reorder each group into
        # canonical llm_names order, drop the rest (reported below).
        self.data_df = self._enforce_rectangular_query_blocks(self.data_df)

        # Optional: exclude queries with zero correctness spread (all 7 models
        # agree, right or wrong) -- their utility label is driven purely by
        # cost/latency/reliability, not skill differentiation. Off by default;
        # set `exclude_degenerate_queries: true` in config to try it, e.g. if
        # you want to concentrate the (already small) training set on the
        # queries that actually discriminate between models.
        if config.get('exclude_degenerate_queries', False):
            group_col = "query_id" if "query_id" in self.data_df.columns else "query"
            degenerate = fb.find_degenerate_queries(self.data_df, group_col)
            if degenerate:
                n_before = self.data_df[group_col].nunique()
                self.data_df = self.data_df[~self.data_df[group_col].isin(degenerate)].reset_index(drop=True)
                print(f"[data_integrity] exclude_degenerate_queries: dropped {len(degenerate)} "
                      f"all-correct/all-wrong queries out of {n_before}.")

        self.num_query = int(len(self.data_df) / self.num_llms)
        self.num_task = config['num_task']
        self.set_seed(self.config['seed'])

        # --- Revised Feature Plan: join router_data with benchmark metadata on
        # Gold_Answer <-> answer (confirmed key), then fit encoders ONLY on the
        # train-split rows -- computed here, before any fitting, so val/test
        # rows cannot leak into one-hot categories or normalization stats. ---
        benchmark_path = benchmark_path or config.get('benchmark_path')
        if benchmark_path and os.path.exists(benchmark_path):
            benchmark_df = pd.read_csv(benchmark_path)
            self.data_df, join_report = fb.join_router_and_benchmark(self.data_df, benchmark_df)
            print(f"[feature_builder] benchmark join match rate: {join_report.match_rate:.3f} "
                  f"({join_report.n_matched}/{join_report.n_router_rows})")
            if join_report.n_ambiguous_benchmark_keys > 0:
                print(f"[feature_builder][WARN] {join_report.n_ambiguous_benchmark_keys} question(s) appear "
                      f"more than once in the benchmark file -- first occurrence used for the join, "
                      f"duplicates dropped.")
            if join_report.match_rate < 0.99:
                print("[feature_builder][WARN] join coverage < 99% -- unmatched rows will get "
                      "imputed/zeroed metadata.")
        else:
            print("[feature_builder][WARN] no benchmark_path provided/found -- proceeding with "
                  "router_data columns only. Domain/Reasoning_Type/Difficulty/Reasoning_Depth/"
                  "Solution_Steps node features will be all-zero.")

        # inference mode operates on a single query slice of an already-trained
        # setup and has no train/val/test split of its own to fit encoders on;
        # it must reuse the SAME split (and therefore same encoders) the model
        # was trained with, so we always compute the canonical split here.
        self._compute_split_row_indices()

        difficulty_order = config.get('difficulty_ordinal_order')  # e.g. ["easy","medium","hard"] or None
        # CONFIRMED from real data: difficulty and reasoning_depth both mix
        # numeric strings ('7','8','9'/'3','4','5') with text labels
        # ('Medium','Very High',...) -- no ordinal mapping supplied by default,
        # both treated as categorical unless *_ordinal_order is set in config.
        reasoning_depth_order = config.get('reasoning_depth_ordinal_order')
        self.utility_weights = resolve_utility_weights(config)
        self.utility_weights = auto_drop_zero_variance_utility_weights(
            self.data_df, self.utility_weights)
        train_df = self.data_df.iloc[self.train_row_idx]
        self.encoders = fb.fit_encoders(
            train_df,
            difficulty_ordinal_order=difficulty_order,
            reasoning_depth_ordinal_order=reasoning_depth_order,
        )
        # in_edges = utility scalar (1) + Phase-4 edge feature vector
        self.edge_dim = 1 + self.encoders.edge_feature_dim
        print(f"[feature_builder] fit on {len(train_df)} train rows -- "
              f"node_metadata_dim={self.encoders.node_metadata_dim}, "
              f"edge_feature_dim={self.encoders.edge_feature_dim} (+1 utility col = {self.edge_dim} total)")

        # DIAGNOSTIC: if node_metadata_dim doesn't drop after mapping difficulty/
        # reasoning_depth to integers, this block tells you exactly why --
        # either the numeric-detection didn't fire (shows raw unique values +
        # frac_numeric), or a stale feature_builder.py without the numeric-
        # detection code entirely (in which case FeatureEncoders won't even
        # have a difficulty_is_numeric attribute, and this print itself errors --
        # that error is itself diagnostic).
        for col_name, is_numeric_attr, categories_attr in [
            ("difficulty", "difficulty_is_numeric", "difficulty_categories"),
            ("reasoning_depth", "reasoning_depth_is_numeric", "reasoning_depth_categories"),
        ]:
            is_numeric = getattr(self.encoders, is_numeric_attr)
            categories = getattr(self.encoders, categories_attr)
            if col_name in train_df.columns:
                raw_uniques = train_df[col_name].dropna().unique().tolist()
                numeric_coerced = pd.to_numeric(train_df[col_name], errors="coerce")
                frac_numeric = numeric_coerced.notna().mean() if len(train_df) else float('nan')
                print(f"[feature_builder][diag] {col_name}: is_numeric={is_numeric}, "
                      f"frac_numeric={frac_numeric:.3f}, dtype={train_df[col_name].dtype}, "
                      f"raw_unique_sample={sorted(map(str, raw_uniques))[:15]}"
                      f"{', categories=' + str(categories[:15]) if not is_numeric else ''}")
            else:
                print(f"[feature_builder][diag] {col_name}: COLUMN NOT PRESENT in joined train_df "
                      f"-- benchmark join likely failed for this column, check benchmark_path and "
                      f"the join match rate printed above.")

        if inference:
            self.num_llms

            start = int(query_id) * self.num_llms
            rows = self.data_df.iloc[start:start + self.num_llms]
            print(f"\n================Task: {rows.iloc[0]['row_id']}================")

            # Phase 4/6: edge features + utility for this query's num_llms rows,
            # built with the SAME encoders fit on the full dataset in __init__
            # (category sets / normalization stats must match training).
            self.edge_features = fb.build_edge_features(rows, self.encoders)
            self.utility_list = fb.compute_utility(rows, self.encoders, **self.utility_weights)

            query_embedding = rows.iloc[0]['query_embedding']
            task_embedding = rows.iloc[0]['task_description_embedding']

            query_embedding = ensure_2d(parse_embedding_field(query_embedding))
            task_embedding = ensure_2d(parse_embedding_field(task_embedding))

            query_embedding = np.array(query_embedding)
            task_embedding = np.array(task_embedding)

            node_metadata = fb.build_node_metadata(rows.iloc[[0]], self.encoders)
            if node_metadata.shape[1] > 0:
                query_embedding = np.concatenate([query_embedding, node_metadata], axis=1)

            query_dim = query_embedding.shape[1]

            self.model = EncoderDecoderNet(query_feature_dim=query_dim, llm_feature_dim=self.llm_dim,
                                           hidden_features=self.config['embedding_dim'], in_edges=self.edge_dim).to(
                device)
            self.form_data = form_data(device)
            results = self.infer_single_query(query_embedding, task_embedding,
                                              query_row_id=rows.iloc[0]['row_id'], rows_df=rows)

            outpath = f"infer_results/{rows.iloc[0]['row_id']}"
            make_plot(results['scores'], output_dir=outpath,
                      scenario=None,
                      query_id=query_id, task_id=rows.iloc[0]['row_id'],
                      utility_weights=self.utility_weights,
                      ranking_metrics=results.get('ranking_metrics'))

            if 'ground_truth_ranking' in results:
                plot_ground_truth_ranking(
                    ground_truth_df=results['ground_truth_ranking'],
                    output_dir=outpath, query_id=query_id, task_id=rows.iloc[0]['row_id'],
                    ranking_metrics=results.get('ranking_metrics'))

            if adaptive:
                feedback_path = f"{self.config['data_dir']}/feedback.json"
                feedback_score = ask_and_save_feedback(query_id, results['best_llm'], feedback_path)

                from data_processing.delayed_reward import AdaptiveBatchUpdater

                adaptive_updater = AdaptiveBatchUpdater(
                    router_csv_path=router_data_path,
                    llm_description_path=llm_path,
                    batch_size=10,
                    eta=0.05,
                    use_feedback=True,
                    beta=0.3
                )

                adaptive_updater.compute(rows.iloc[0]['query'],
                                         feedback_score,
                                         results['best_llm'],
                                         query_id,
                                         self.num_llms)

                adaptive_updater.flush()




        elif config.get('kfold_cv', False):
            # K-Fold CV mode: bypasses the single 90/10-style split entirely
            # (query_dim/edge_dim still need self.prepare_data_for_GNN(),
            # but self.split_data()'s single train/val/test masks are unused
            # here -- run_kfold_cv builds its own per-fold masks).
            self.prepare_data_for_GNN()
            self.query_dim = self.query_embedding_list.shape[1]

            # Same per-query softmax + one-hot label derivation split_data()
            # does -- independent of any train/val/test split, so it's safe
            # to compute here without calling split_data() itself.
            SOFTMAX_TEMPERATURE = 1.0
            utility_reshaped = self.utility_list.reshape(-1, self.num_llms) / SOFTMAX_TEMPERATURE
            exp_scores = np.exp(utility_reshaped - np.max(utility_reshaped, axis=1, keepdims=True))
            softmax_per_query = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            utility_softmaxed = softmax_per_query.reshape(-1)
            utility_re = utility_softmaxed.reshape(-1, self.num_llms)
            kfold_label = np.eye(self.num_llms)[np.argmax(utility_re, axis=1)].reshape(-1, 1)

            # Router-level evaluation (Relative Utility Gap Closed vs Oracle/
            # SBM/Random/Cost-Optimal baselines, Top-1/Top-K match, NDCG@3)
            # needs the RAW (pre-softmax) utility per query-model edge
            # attached to data_df, since that's what "which model was
            # actually best for this query" has to be judged against --
            # the softmaxed/one-hot label is a training target, not a
            # utility scale. self.utility_list at this point IS that raw
            # scale (fb.compute_utility output, before the softmax block
            # above reassigns a local `utility_softmaxed`).
            eval_df = self.data_df.copy()
            eval_df['utility'] = self.utility_list

            self.kfold_results = run_kfold_cv(
                query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim,
                hidden_features_size=self.config['embedding_dim'], in_edges_size=self.edge_dim,
                config=self.config, device=device,
                task_embedding_list=self.task_embedding_list,
                query_embedding_list=self.query_embedding_list,
                llm_description_embedding=self.llm_description_embedding,
                edge_org_id=[num for num in range(self.num_query) for _ in range(self.num_llms)],
                edge_des_id=list(range(self.num_llms)) * self.num_query,
                utility_list=utility_softmaxed, label=kfold_label, combined_edge=self.edge_features,
                num_llms=self.num_llms, num_query=self.num_query,
                k=config.get('kfold_k', 5),
                inner_val_ratio=config.get('kfold_inner_val_ratio', 0.1),
                wandb=self.wandb, make_plots=config.get('kfold_make_plots', False),
                data_df=eval_df, gold_col='utility',
                cost_col=config.get('router_eval_cost_col', 'Cost'),
                correctness_col=config.get('router_eval_correctness_col', 'Correct'),
                latency_col=config.get('router_eval_latency_col', 'Latency'),
            )
        else:
            from graph_nn import GNN_prediction
            self.prepare_data_for_GNN()
            self.split_data()
            self.form_data = form_data(device)
            self.query_dim = self.query_embedding_list.shape[1]
            self.GNN_predict = GNN_prediction(query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim,
                                              hidden_features_size=self.config['embedding_dim'],
                                              in_edges_size=self.edge_dim, wandb=self.wandb, config=self.config,
                                              device=device)
            print("GNN training successfully initialized.")
            self.train_GNN()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _compute_split_row_indices(self):
        """
        Row indices (into self.data_df, 0-indexed, task/query/llm-major order)
        for train/validate/test. Extracted out of split_data() so it can run
        BEFORE encoder fitting -- encoders must only see train rows.
        Idempotent: safe to call more than once (e.g. once early for encoder
        fitting, once again implicitly via split_data() for mask construction).
        """
        self.query_per_task = int(self.num_query / self.num_task)
        split_ratio = self.config['split_ratio']

        train_size = int(self.query_per_task * split_ratio[0])
        val_size = int(self.query_per_task * split_ratio[1])
        test_size = int(self.query_per_task * split_ratio[2])

        train_idx, validate_idx, test_idx = [], [], []
        for task_id in range(self.num_task):
            start_idx = task_id * self.query_per_task * self.num_llms
            train_idx.extend(range(start_idx, start_idx + train_size * self.num_llms))
            validate_idx.extend(range(start_idx + train_size * self.num_llms,
                                      start_idx + train_size * self.num_llms + val_size * self.num_llms))
            test_idx.extend(range(start_idx + train_size * self.num_llms + val_size * self.num_llms,
                                  start_idx + train_size * self.num_llms + val_size * self.num_llms + test_size * self.num_llms))

        self.train_row_idx = train_idx
        self.validate_row_idx = validate_idx
        self.test_row_idx = test_idx
        return train_idx, validate_idx, test_idx

    def _enforce_rectangular_query_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantee the invariant every downstream positional index relies on:
        each query occupies exactly self.num_llms consecutive rows, one per
        distinct model value found in the router data. Groups by `query_id`
        (present in router_data.csv per Revised_Feature_Plan.md); falls back
        to grouping by raw `query` text with a warning if query_id is missing.

        The "complete set" of models is derived from the router dataset's own
        `model` column (self-consistency check), NOT compared against
        LLM_Descriptions.json's llm_names -- the two use unrelated naming
        schemes and matching them is a separate concern from data integrity.
        The one thing we do verify is that the router data's own model
        vocabulary has exactly self.num_llms distinct values, since that's
        the cardinality every downstream tensor shape assumes.

        Any query_id whose row set doesn't cover that vocabulary exactly once
        (missing run, duplicate run) is DROPPED WHOLESALE and reported --
        keeping it would silently misalign edge_org_id/edge_des_id/
        unique_index_list for every query after it, which is strictly worse
        than losing that one query's data.
        """
        group_col = "query_id" if "query_id" in df.columns else "query"
        if group_col == "query":
            print("[data_integrity][WARN] no query_id column found -- grouping by raw "
                  "query text instead. If two different queries have identical text this "
                  "will incorrectly merge them.")

        wanted_models = sorted(df["model"].unique().tolist())
        if len(wanted_models) != self.num_llms:
            raise ValueError(
                f"[data_integrity] router_data's `model` column has {len(wanted_models)} distinct "
                f"values {wanted_models}, but LLM_Descriptions.json declares {self.num_llms} LLMs "
                f"({self.llm_names}). These counts must match -- check for typos/extra models in "
                f"the data or a stale LLM_Descriptions.json."
            )
        wanted_models_set = set(wanted_models)
        llm_order = {name: i for i, name in enumerate(wanted_models)}  # stable order, data-derived

        # BUGFIX (name mismatch): `wanted_models` (raw router ids, e.g.
        # 'openai/gpt-5') is what every row gets positionally sorted into,
        # and self.llm_names (display names, e.g. 'GPT-5') is what the rest
        # of the pipeline (scores dict, best_llm, etc.) is keyed by. Nothing
        # upstream ever recorded which raw id corresponds to which display
        # name -- the two lists just happen to share the same positional
        # order once both are the index-i-th model. Persist that
        # correspondence here, once, so downstream code (infer_single_query /
        # compute_ranking_metrics) can translate router ids <-> display
        # names instead of comparing them directly.
        self.router_model_ids = wanted_models
        self.router_id_to_display = dict(zip(self.router_model_ids, self.llm_names))

        kept_blocks = []
        n_dropped_queries = 0
        dropped_ids = []
        for key, group in df.groupby(group_col, sort=False):
            models_here = group["model"].tolist()
            if len(models_here) != self.num_llms or set(models_here) != wanted_models_set:
                n_dropped_queries += 1
                dropped_ids.append(key)
                continue
            kept_blocks.append(group.assign(_llm_sort=group["model"].map(llm_order)).sort_values("_llm_sort"))

        if n_dropped_queries:
            preview = dropped_ids[:10]
            print(f"[data_integrity][WARN] dropped {n_dropped_queries} incomplete/duplicate "
                  f"query group(s) out of {df[group_col].nunique()} -- these queries did not "
                  f"have exactly one row per model in {wanted_models}. Example query_id(s): {preview}"
                  f"{' ...' if n_dropped_queries > 10 else ''}")

        if not kept_blocks:
            raise ValueError(
                "[data_integrity] 0 complete query blocks after filtering -- check the `model` "
                "column values in router_data.csv for typos/inconsistent naming: " + str(wanted_models)
            )

        result = pd.concat(kept_blocks, axis=0, ignore_index=True).drop(columns=["_llm_sort"])
        n_before, n_after = len(df), len(result)
        print(f"[data_integrity] rectangular-block check: {n_before} -> {n_after} rows "
              f"({n_before - n_after} dropped), {n_after // self.num_llms} complete queries retained.")
        return result

    def split_data(self):
        train_idx, validate_idx, test_idx = self._compute_split_row_indices()

        # --- Phase 6 (revised): label + edge tensor derived from the utility score and
        # the Phase-4 edge feature vector, replacing the old cost/effect scenario-softmax. ---
        # combined_edge = the full Phase-4 feature vector (Correct, Cost_norm, Latency_norm,
        # Input_Tokens_norm, Output_Tokens_norm, Completion_Status_onehot, Error_Type_onehot).
        # form_data.formulation() will prepend the utility scalar as an additional column,
        # so total edge width fed to the GNN is (1 + edge_feature_dim) == self.edge_dim.
        self.combined_edge = self.edge_features

        # softmax-per-query over utility, kept as a smoothed ranking signal for the
        # test()/train_validate() top-1 comparisons in graph_nn.py (which read edge_attr
        # as a per-query distribution, not a raw scalar).
        # NOTE: previously SOFTMAX_TEMPERATURE = 0.0001, which multiplies utility
        # gaps (~0.1-1.5 in practice) by 10,000x before exponentiating -- this
        # collapses the "smoothed" ground-truth distribution to a near-hard
        # one-hot regardless of how close the actual utilities are, and trains
        # the model against labels that don't reflect true tie-closeness.
        # T=1.0 is plain softmax over the raw utility scale; tune from there
        # (values <1 sharpen, >1 flatten) if you want more/less separation,
        # but don't drop back to ~1e-4.
        SOFTMAX_TEMPERATURE = 1.0
        utility_reshaped = self.utility_list.reshape(-1, self.num_llms) / SOFTMAX_TEMPERATURE
        exp_scores = np.exp(utility_reshaped - np.max(utility_reshaped, axis=1, keepdims=True))
        softmax_per_query = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.utility_list = softmax_per_query.reshape(-1)

        utility_re = self.utility_list.reshape(-1, self.num_llms)
        self.label = np.eye(self.num_llms)[np.argmax(utility_re, axis=1)].reshape(-1, 1)
        self.edge_org_id = [num for num in range(self.num_query) for _ in range(self.num_llms)]
        self.edge_des_id = list(range(self.edge_org_id[0], self.edge_org_id[0] + self.num_llms)) * self.num_query

        self.mask_train = torch.zeros(len(self.edge_org_id))
        self.mask_train[train_idx] = 1

        self.mask_validate = torch.zeros(len(self.edge_org_id))
        self.mask_validate[validate_idx] = 1

        self.mask_test = torch.zeros(len(self.edge_org_id))
        self.mask_test[test_idx] = 1

    def prepare_data_for_GNN(self):
        unique_index_list = list(range(0, len(self.data_df), self.num_llms))
        query_embedding_list_raw = self.data_df['query_embedding'].tolist()
        task_embedding_list_raw = self.data_df['task_description_embedding'].tolist()

        self.query_embedding_list = []
        self.task_embedding_list = []
        for inter in query_embedding_list_raw:
            q_emb = parse_embedding_field(inter)
            self.query_embedding_list.append(q_emb)

        for inter in task_embedding_list_raw:
            t_emb = parse_embedding_field(inter)
            self.task_embedding_list.append(t_emb)

        self.query_embedding_list = np.array(self.query_embedding_list)[unique_index_list]
        self.task_embedding_list = np.array(self.task_embedding_list)[unique_index_list]

        # --- Phase 2/3: Question Representation = MLP(Concat(Text_Embedding, Metadata_Vector)) ---
        # The "MLP" half is FeatureAlign.query_transform inside graph_nn.py (unchanged);
        # here we build the "Concat" half by appending the metadata vector to the raw
        # text embedding, subset to one row per unique query exactly as the embeddings above.
        node_metadata = fb.build_node_metadata(self.data_df, self.encoders)[unique_index_list]
        if node_metadata.shape[1] > 0:
            self.query_embedding_list = np.concatenate(
                [self.query_embedding_list, node_metadata], axis=1
            )

        # --- Phase 4: edge feature vector (per router_data row, i.e. per query-model edge) ---
        self.edge_features = fb.build_edge_features(self.data_df, self.encoders)

        # --- Phase 6: usable-today utility score, used both as GNN training label source
        # and as the ground-truth ranking signal (replaces the old cost/effect-scenario logic) ---
        self.utility_list = fb.compute_utility(self.data_df, self.encoders, **self.utility_weights)

    def train_GNN(self):

        self.data_for_GNN_train = self.form_data.formulation(task_id=self.task_embedding_list,
                                                             query_feature=self.query_embedding_list,
                                                             llm_feature=self.llm_description_embedding,
                                                             org_node=self.edge_org_id,
                                                             des_node=self.edge_des_id,
                                                             edge_feature=self.utility_list, edge_mask=self.mask_train,
                                                             label=self.label, combined_edge=self.combined_edge,
                                                             train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                             test_mask=self.mask_test)
        self.data_for_GNN_validate = self.form_data.formulation(task_id=self.task_embedding_list,
                                                                query_feature=self.query_embedding_list,
                                                                llm_feature=self.llm_description_embedding,
                                                                org_node=self.edge_org_id,
                                                                des_node=self.edge_des_id,
                                                                edge_feature=self.utility_list,
                                                                edge_mask=self.mask_validate, label=self.label,
                                                                combined_edge=self.combined_edge,
                                                                train_mask=self.mask_train,
                                                                valide_mask=self.mask_validate,
                                                                test_mask=self.mask_test)

        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.utility_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test)
        self.GNN_predict.train_validate(data=self.data_for_GNN_train, data_validate=self.data_for_GNN_validate,
                                        data_for_test=self.data_for_test)

    def test_GNN(self):
        predicted_result = self.GNN_predict.test(data=self.data_for_test, model_path=self.config['model_path'])

    def infer_single_query(self, query_embedding, task_embedding=None, query_row_id=None, rows_df=None):
        """
        Perform inference for ONE query.

        Args:
            query_embedding: numpy array (query_dim,)
            task_embedding: numpy array (task_dim,) (if None, use first task)
            query_row_id: the query's 'row_id' value in the router_data
                DataFrame -- used to compute the ground-truth ranking via
                calculate_query_utility(). If None, ranking metrics/plot are
                skipped (predicted scores are still returned as before).
            rows_df: the num_llms-row slice of router_data.csv for this
                query (must contain 'model', 'Correct', 'Cost', 'Latency',
                'Output_Tokens', 'Completion_Status'). Required alongside
                query_row_id to compute ranking metrics.

        Returns:
            dict with scores, best llm, and (if rows_df/query_row_id given)
            'ranking_metrics' and 'ground_truth_ranking'.
        """

        # Load best model
        # BUGFIX: graph_nn.py's GNN_prediction now treats config['model_path']
        # as a DIRECTORY (it saves to {model_path}/best_model.pth), not a file
        # path -- config['model_path'] must be a directory now (e.g.
        # "checkpoints/"), not "checkpoints/best_model.pth" like before.
        checkpoint_path = os.path.join(self.config['model_path'], "best_model.pth")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        # If only one task in your setup
        if task_embedding is None:
            task_embedding = self.task_embedding_list[0]

        query_embedding = np.array(query_embedding).reshape(1, -1)
        task_embedding = np.array(task_embedding).reshape(1, -1)

        num_llms = self.num_llms

        # ---- Construct Graph ----

        # Query → LLM edges
        org_node = [0] * num_llms
        des_node = list(range(num_llms))  # will be shifted inside formulation

        # combined_edge: Phase-4 edge feature vector for this query's num_llms rows,
        # already built in __init__ using the encoders fit on the full dataset.
        combined_edge = self.edge_features

        # Smoothed per-query utility distribution, used as the ground-truth ranking
        # signal below (data.edge_attr) -- same treatment as split_data() at train time.
        # Must match split_data()'s temperature exactly -- otherwise the GT
        # distribution printed here is not comparable to what the model was
        # trained against.
        SOFTMAX_TEMPERATURE = 1.0
        utility_reshaped = self.utility_list.reshape(-1, self.num_llms) / SOFTMAX_TEMPERATURE
        exp_scores = np.exp(utility_reshaped - np.max(utility_reshaped, axis=1, keepdims=True))
        softmax_per_query = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.utility_list = softmax_per_query.reshape(-1)

        # Masks (we predict ALL edges)
        edge_mask = torch.ones(num_llms).to(device)
        train_mask = torch.ones(num_llms).to(device)
        validate_mask = torch.zeros(num_llms).to(device)
        test_mask = torch.zeros(num_llms).to(device)

        # Dummy label (not used)
        dummy_label = np.zeros((num_llms, 1))

        # ---- Build Data ----
        data = self.form_data.formulation(
            task_id=task_embedding,
            query_feature=query_embedding,
            llm_feature=self.llm_description_embedding,
            org_node=org_node,
            des_node=des_node,
            edge_feature=self.utility_list,
            label=dummy_label,
            edge_mask=edge_mask,
            combined_edge=combined_edge,
            train_mask=train_mask,
            valide_mask=validate_mask,
            test_mask=test_mask
        )

        # During inference, allow full visibility
        edge_can_see = torch.ones(num_llms).bool().to(device)

        # ---- Forward ----
        with torch.no_grad():
            edge_scores = self.model(
                task_id=data.task_id,
                query_features=data.query_features,
                llm_features=data.llm_features,
                edge_index=data.edge_index,
                edge_mask=edge_mask.bool(),
                edge_can_see=edge_can_see,
                edge_weight=data.combined_edge
            )
        pred = edge_scores.reshape(-1, self.num_llms)
        # NOTE: previously T = .000001 -- an independent, far more extreme
        # sharpening stacked on top of whatever calibrated distribution the
        # model already produced. This crushed any close call to a hard
        # 1.0/0.0/0.0 even when the model's raw logits were only mildly
        # separated (example 2 in the bug report: real logit gap ~2.8 was
        # printed as 0.9396/0.0604/0.0000 instead of something closer to the
        # model's actual confidence). T=1.0 reports the model's output as-is;
        # if edge_scores are already probabilities (model has a softmax head),
        # skip this softmax entirely instead of reapplying one.
        T = 1.0
        pred = torch.softmax(pred / T, dim=1)

        # edge_scores = edge_scores.cpu().numpy().reshape(-1)
        best_idx = torch.argmax(pred, 1)

        gt_scores = data.edge_attr.reshape(-1, self.num_llms)
        gt_idx = torch.argmax(gt_scores, 1)

        # Top-3 predicted indices
        edge_scores = pred[0].cpu().numpy()
        top3_pred_idx = np.argsort(edge_scores)[::-1][:3]
        top3_pred_scores = edge_scores[top3_pred_idx]

        # If GT is one-hot per query:
        gt_scores = gt_scores[0].cpu().numpy()
        top3_gt_idx = np.argsort(gt_scores)[::-1][:3]
        top3_gt_scores = gt_scores[top3_gt_idx]

        print("Top-3 Predicted LLMs:")
        for rank, (idx, score) in enumerate(zip(top3_pred_idx, top3_pred_scores), 1):
            print(f"{rank}. LLM {self.llm_names[idx]:<3} | Score: {score:.4f}")

        print("\nTop-3 Ground Truth LLMs:")
        for rank, (idx, score) in enumerate(zip(top3_gt_idx, top3_gt_scores), 1):
            print(f"{rank}. LLM {self.llm_names[idx]:<3} | Score: {score:.4f}")

        print("\n--------------------------------------------")

        gt_scores = [round(float(s), 3) for s in gt_scores]

        scores = {
            self.llm_names[i]: float(pred[0, i].cpu().item())
            for i in range(len(self.llm_names))
        }

        result = {
            "best_llm": self.llm_names[best_idx],
            "scores": scores,
            "edge_scores": edge_scores,
            'ground_truth': self.llm_names[gt_idx],
            'gt_scores': gt_scores
        }

        # --- Metrics between predicted ranking and ground-truth ranking ---
        # Uses calculate_query_utility() (raw Correct/Cost/Latency/Output_Tokens/
        # Completion_Status columns, NOT the softmax-smoothed edge_attr used
        # above) as the ground truth, since that's the actual utility scale
        # the router is meant to be judged against.
        if query_row_id is not None and rows_df is not None:
            # try:
            rows_with_qid = rows_df.assign(query_id=query_row_id)
            ground_truth_ranking = calculate_query_utility(
                query_id=query_row_id, df=rows_with_qid,
                weights=self.utility_weights, query_id_col='query_id')

            # BUGFIX (name mismatch): calculate_query_utility()'s 'model' column
            # comes straight from rows_df['model'], i.e. raw router ids
            # ('openai/gpt-5'), while `scores` is keyed by display names
            # ('GPT-5') via self.llm_names. Translate the ground-truth model
            # column to display names (using the positional correspondence
            # recorded in self.router_id_to_display by
            # _enforce_rectangular_query_blocks) so both sides use the same
            # naming scheme for compute_ranking_metrics below.
            unmapped = set(ground_truth_ranking['model']) - set(self.router_id_to_display)
            if unmapped:
                raise ValueError(
                    f"compute_ranking_metrics: router id(s) {sorted(unmapped)} in the ground-truth "
                    f"'model' column have no entry in router_id_to_display -- "
                    f"router_model_ids={self.router_model_ids} vs llm_names={self.llm_names} "
                    f"are out of sync."
                )
            ground_truth_ranking['model'] = ground_truth_ranking['model'].map(self.router_id_to_display)

            ranking_metrics = compute_ranking_metrics(scores, ground_truth_ranking)
            print("\n=== Predicted vs Ground-Truth Ranking Metrics ===")
            print(f"Top-1 Match          : {ranking_metrics['top1_match']}")
            print(f"Spearman Rank Corr   : {ranking_metrics['spearman_rank_corr']:.4f}")
            print(f"NDCG@3               : {ranking_metrics['ndcg_at_3']:.4f}")
            print(f"Mean Abs Rank Error  : {ranking_metrics['mean_abs_rank_error']:.4f}")
            result['ranking_metrics'] = ranking_metrics
            result['ground_truth_ranking'] = ground_truth_ranking
            # except (KeyError, ValueError) as e:
            #     print(f"[ranking_metrics][WARN] skipped -- {e}")

        return result


# if __name__ == "__main__":
#     import wandb
#
#     with open("configs/config.yaml", 'r', encoding='utf-8') as file:
#         config = yaml.safe_load(file)
#     wandb_key = config['wandb_key']
#     wandb.login(key=wandb_key)
#     wandb.init(project="graph_router")
#
#     data_dir = config['data_dir']
#     router_data_path = os.path.join(data_dir, 'router_data.csv')
#     if config['feedback']:
#         router_data_path = os.path.join(data_dir, 'feedback/router_data.csv')
#         if not os.path.exists(router_data_path):
#             print("[INFO] No feedback found")
#             router_data_path = os.path.join(data_dir, 'router_data.csv')
#
#     graph_router_prediction(
#
#         router_data_path=router_data_path,
#         llm_path=os.path.join(data_dir, 'LLM_Descriptions.json'),
#         llm_embedding_path=os.path.join(data_dir, "llm_description_embedding.pkl"),
#         config=config,
#         wandb=wandb
#     )