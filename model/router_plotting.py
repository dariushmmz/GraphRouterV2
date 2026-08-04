"""
Plotly chart generation for router predictions.

Extracted from multi_task_graph_router.py -- no logic changes, only
relocation. Depends only on plain dicts/DataFrames passed in, not on
graph_router_prediction's internal state.
"""

import os

import pandas as pd


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

    except Exception:
        pio.renderers.default = "colab"

    fig.show()
