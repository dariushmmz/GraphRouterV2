"""
Lightweight explainability for the graph router's model-selection decision.

Design goals (deliberately minimal):
  - No SHAP, no gradient/attention attribution.
  - Reuses the SAME decision-layer signals already computed elsewhere in the
    pipeline (predicted score, and -- when available -- the raw Correct /
    Cost / Latency / Completion_Status / Output_Tokens columns and the
    utility_weights that already score them in router_utility.py).
  - Pure functions, no dependency on graph_router_prediction's internal
    state, so this can be unit-tested and reused independently -- same
    pattern as router_utility.py.

Two tiers, chosen automatically based on what data is available at the
call site:

  1. Rich tier (rows_df provided): ranks Correct / Cost / Latency /
     Completion_Status / Output_Tokens by each one's WEIGHTED CONTRIBUTION
     gap between the selected model and the overall runner-up, using the
     exact utility_weights already in play. This is only possible when a
     historical/ground-truth slice for the query exists (e.g. eval-time
     inference against router_data.csv), since raw Cost/Latency/outcome
     for a brand-new, never-run query is not observable ahead of time.

  2. Fallback tier (rows_df is None -- true blind inference on an unseen
     query): only the predicted-score margin over the runner-up is
     observable, so the explanation is built from that alone.

Either way the output format is fixed:

    Selected Model: <model_name>
    Reasons:
      - <reason 1>
      - <reason 2>
      - <reason 3>
"""

import numpy as np
import pandas as pd

# criterion_key -> (utility weight key, raw column, higher_raw_is_better,
#                    strong-phrasing template, mild-phrasing template)
# strong template is used when the selected model is the BEST candidate on
# that criterion; mild template is used when it's not the best but also not
# the worst (i.e. "acceptable"). Templates with {runner_up} get the overall
# runner-up model's display name interpolated in.
_CRITERIA_SPEC = {
    'success': dict(
        weight_key='w_success', column='Correct', higher_is_better=True,
        strong="High predicted success for this query",
        mild="Reasonable predicted success for this query",
    ),
    'reliability': dict(
        weight_key='w_completion_reliability', column='Completion_Status', higher_is_better=True,
        strong="High completion reliability",
        mild="Acceptable completion reliability",
    ),
    'cost': dict(
        weight_key='w_cost', column='Cost', higher_is_better=False,
        strong="Lower estimated cost than {runner_up}",
        mild="Acceptable cost",
    ),
    'latency': dict(
        weight_key='w_latency', column='Latency', higher_is_better=False,
        strong="Lower latency than {runner_up}",
        mild="Acceptable latency",
    ),
    'output_tokens': dict(
        weight_key='w_output_tokens', column='Output_Tokens', higher_is_better=False,
        strong="More concise output than {runner_up}",
        mild="Reasonable output length",
    ),
}


def _reliability_series(raw: pd.Series) -> pd.Series:
    """Same Completion_Status -> {0,1}-ish mapping used in router_utility.py."""
    return raw.apply(
        lambda x: 1.0 if x in [1, True, 'Success', 'SUCCESS', 'Completed', 'completed']
        else (float(x) if str(x).replace('.', '', 1).isdigit() else 0.0)
    )


def _min_max_norm(series: pd.Series) -> pd.Series:
    rng = series.max() - series.min()
    return (series - series.min()) / rng if rng > 0 else pd.Series(0.0, index=series.index)


def _confidence_signal(selected_model: str, scores: dict, runner_up: str | None) -> tuple:
    """Predicted-score-margin signal -- always available, used as the
    fallback tier and also as a filler when the rich tier can't fill all 3
    slots."""
    selected_score = scores[selected_model]
    if runner_up is None:
        reason = f"Highest predicted routing confidence ({selected_score:.0%})"
        gap = selected_score
    else:
        runner_up_score = scores[runner_up]
        gap = selected_score - runner_up_score
        reason = (
            f"Highest predicted routing confidence "
            f"({selected_score:.0%} vs {runner_up_score:.0%} for {runner_up})"
        )
    return ('confidence', reason, gap)


def _criteria_signals(
    selected_model: str,
    runner_up: str | None,
    weights: dict,
    rows_df: pd.DataFrame,
    model_col: str,
    id_to_display: dict | None,
) -> list:
    """Rich-tier signals: one candidate signal per active (non-zero-weight)
    criterion, keyed off the exact weighted-utility formula already used
    for training/ground-truth ranking."""
    df = rows_df.copy()
    if id_to_display:
        df[model_col] = df[model_col].map(id_to_display).fillna(df[model_col])
    df = df.set_index(model_col)

    signals = []
    for key, spec in _CRITERIA_SPEC.items():
        weight = weights.get(spec['weight_key'], 0) or 0
        column = spec['column']
        if weight == 0 or column not in df.columns:
            continue
        if selected_model not in df.index:
            continue

        raw = df[column].astype(float) if column != 'Completion_Status' else _reliability_series(df[column])

        if spec['higher_is_better']:
            normalized = raw if column == 'Correct' or column == 'Completion_Status' else _min_max_norm(raw)
            contribution = weight * normalized
        else:
            normalized = _min_max_norm(raw)
            contribution = -weight * normalized

        selected_contribution = contribution.get(selected_model, np.nan)
        if pd.isna(selected_contribution):
            continue

        best_model = contribution.idxmax()
        is_best = (best_model == selected_model)

        # gap vs the overall runner-up (fall back to gap vs the field's mean
        # if there's no runner-up, e.g. only one candidate model).
        if runner_up is not None and runner_up in contribution.index:
            gap = selected_contribution - contribution[runner_up]
        else:
            gap = selected_contribution - contribution.drop(selected_model, errors='ignore').mean()

        # "worst" candidates aren't cited as reasons at all -- only best
        # (strong) or middle-of-the-pack (mild, i.e. not the worst).
        worst_model = contribution.idxmin()
        if selected_model == worst_model and len(contribution) > 1:
            continue

        template = spec['strong'] if is_best else spec['mild']
        reason = template.format(runner_up=runner_up) if '{runner_up}' in template else template
        signals.append((key, reason, gap))

    return signals


def explain_selection(
    selected_model: str,
    scores: dict,
    weights: dict,
    rows_df: pd.DataFrame = None,
    model_col: str = 'model',
    id_to_display: dict = None,
    top_k: int = 3,
) -> dict:
    """
    Build a short, human-readable explanation for why `selected_model` was
    chosen, using only signals already present in the decision layer.

    Parameters
    ----------
    selected_model : the display-name of the chosen model (matches
        `scores` keys, e.g. self.llm_names entries).
    scores : {model_display_name: predicted_score} for ALL candidates,
        e.g. the `scores` dict already built in infer_single_query.
    weights : the active utility_weights dict (e.g. self.utility_weights)
        -- same weights already used by router_utility.calculate_query_utility.
    rows_df : optional. The num_llms-row slice with raw 'model', 'Correct',
        'Cost', 'Latency', 'Output_Tokens', 'Completion_Status' columns for
        this query (same object already passed into infer_single_query for
        ranking metrics). If None, only the predicted-score margin is used
        (true blind inference on an unseen query has no raw outcome data
        to fall back on).
    model_col : column in rows_df holding the model identifier.
    id_to_display : optional {raw_router_id: display_name} map (e.g.
        self.router_id_to_display) -- needed because rows_df typically uses
        raw router ids while `scores` uses display names.
    top_k : number of reasons to surface (default 3, per spec).

    Returns
    -------
    dict with:
        'text'    : the formatted "Selected Model: ... / Reasons: ..." string
        'signals' : ordered list of (criterion_key, reason_text, gap) for
                    the reasons actually used, best-first -- for callers
                    that want the structured form instead of/alongside text.
    """
    if selected_model not in scores:
        raise ValueError(f"explain_selection: selected_model '{selected_model}' not found in scores keys "
                          f"{list(scores)}.")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    runner_up = next((m for m, _ in ranked if m != selected_model), None)

    candidates = [_confidence_signal(selected_model, scores, runner_up)]
    if rows_df is not None:
        candidates += _criteria_signals(
            selected_model, runner_up, weights, rows_df, model_col, id_to_display
        )

    # Best-first by contribution/margin gap; keep top_k.
    candidates.sort(key=lambda s: s[2], reverse=True)
    top = candidates[:top_k]

    lines = [f"Selected Model: {selected_model}", "Reasons:"]
    lines += [f"  - {reason}" for _, reason, _ in top]
    lines += '\n'
    text = "\n".join(lines)

    return {'text': text, 'signals': top}
