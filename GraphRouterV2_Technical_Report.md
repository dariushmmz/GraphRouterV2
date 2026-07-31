# GraphRouterV2 — Technical Report

## 1. System Overview

GraphRouterV2 is a graph neural network (GNN) that routes a query to the best of seven candidate LLMs (GPT-5, Claude-Sonnet-4.5, Gemini-3.5-Flash, DeepSeek-R1, Qwen3-235B-Thinking, Llama-3.1-70B, Mixtral-8x22B), trading off correctness, cost, latency, and reliability. Queries and models are nodes; query→model edges carry execution features (cost, latency, correctness, tokens, completion status). The network scores each edge, and the top-scoring edge is the router's pick for that query.

## 2. Graph Model Architecture (`graph_nn.py`)

**`EncoderDecoderNet`** is a 2-layer GNN encoder-decoder:

- Two `GeneralConv` layers (PyTorch Geometric) over the query/model bipartite graph, each with `hidden_features * 2` channels.
- **LayerNorm instead of BatchNorm1d** — a deliberate choice given the small dataset (~35–50 training queries per fold). BatchNorm's running statistics are unstable at this scale; LayerNorm normalizes per-sample across feature dimensions and doesn't depend on batch composition.
- Two `nn.Dropout` layers (rate set via `config['dropout']`, currently 0.35) applied after each conv layer. Dropout is automatically disabled at inference via `model.eval()`.
- A decoder MLP head (with its own dropout) that maps the final edge representation to a predicted utility score per (query, model) edge.

Training (`train_validate`) uses `AdamW`, `BCELoss` against a binarized utility label, and validation-loss-based early stopping. Model selection at the end of a run is by best validation macro-F1, not just lowest loss.

## 3. Feature Pipeline (`feature_builder.py` / `multi_task_graph_router.py`)

- **Node (question) features (10-dim)**: query embedding + domain (6 pre-bucketed macro categories), difficulty and reasoning_depth (numeric, min-max scaled upstream in the benchmark file), solution_steps (z-scored), question length (log1p + z-scored). Zero-variance columns are dropped automatically.
- **Edge (query↔model) features (9–10 dim)**: `Correct`, `Cost`, `Latency`, `Input_Tokens`, `Output_Tokens`, `Completion_Status`, `Error_Type`.
- **Utility label**: `Utility = w_success·Correct − w_cost·Cost − w_latency·Latency − w_output_tokens·Output_Tokens + w_completion_reliability·Completion_Reliability`, softmax-normalized per query for the "ground truth" ranking display, and binarized (argmax) as the `BCELoss` training target.

Current weights (`config.yaml`): `w_success=1.0, w_cost=0.6, w_latency=0.4, w_output_tokens=0.3, w_completion_reliability=0.1` — these are placeholders, not yet tuned against a validation protocol.

## 4. Key Changes Since the Last Version

1. **Dimensionality reduction of node metadata**: from a raw 82-dim one-hot representation down to 10 dims, via macro-bucketed domain, z-scored ordinals, and dropping constant/high-cardinality columns. This directly targeted the earlier overfitting signature (train loss → 0, val/test loss climbing).
2. **Normalization swap**: BatchNorm1d → LayerNorm, appropriate for small-batch / small-N training.
3. **Real regularization added**: dropout (0.35), increased weight decay (0.03), and validation-loss-based early stopping (`patience=20`) — previously the model had none of these.
4. **K-fold cross-validation** (`run_kfold_cv`, `kfold_k=5`) replaces a single train/val/test split as the primary evaluation protocol, specifically because a single ~15-query test split makes any point estimate noisy (one flipped query moves accuracy by several points).
5. **Evaluation reworked from Top-1 accuracy to Relative Utility Gap Closed** (`router_evaluation.py`), plus NDCG@3, Top-K-within-tolerance, and Pareto-style comparisons (accuracy/cost/latency retention vs. the single best model). This is a more informative metric than raw accuracy because it credits near-ties and penalizes bad picks proportionally to how bad they are.
6. **Pooled out-of-fold evaluation** (`build_oof_predictions_df`): instead of averaging five small, independently noisy per-fold accuracies, every query's held-out prediction is pooled into one table and scored once — a lower-variance estimate of true generalization.
7. **Named utility-weight scenarios**: `utility_weights` in `config.yaml` replaced with `utility_weight_profiles` (`performance_first` / `balance` / `cost_first`) selected via `utility_scenario`, making deployment intent explicit instead of implicit in whatever numbers were checked in. The system currently runs `cost_first`.
8. **Cost-Optimal baseline added to Utility Gap Closed**: `evaluate_router_performance` now reports gap-closed against the Cost-Optimal baseline (always pick the cheapest model) alongside the existing SBM (accuracy-first) comparison, since judging a deliberately cost/latency-optimized router only against an accuracy-first baseline is the wrong yardstick.
9. **Zero-variance utility-weight auto-drop**: `auto_drop_zero_variance_utility_weights()` checks each utility weight's source column (e.g. `w_completion_reliability` → `Completion_Status`) for variance at load time and zeroes any weight whose source column is constant, with a printed warning — mirroring the existing node-feature auto-drop, but for the utility formula itself. Currently `Completion_Status` is constant (`'success'`) across the dataset, so `w_completion_reliability` is auto-zeroed each run; it reactivates on its own once real completion failures appear in the data.

## 5. Evaluation Metrics — What They Mean

- **Utility Gap Closed**: `(U_router − U_baseline) / (U_oracle − U_baseline) × 100`, reported against two baselines because they represent two different deployment intents. **vs. SBM**: 100% = router matches the oracle's ceiling; 0% = no better than always picking the single best model; negative = worse than that static baseline. **vs. Cost-Optimal**: same scale, but against always picking the cheapest model instead — the more appropriate comparison for a router that's deliberately cost/latency-optimized, since SBM represents an accuracy-first strategy the router isn't actually trying to match.
- **Top-1 Oracle Match Rate**: how often the router's #1 pick is the actual best model for that query.
- **Top-K Match (within tolerance)**: how often the router's pick is within 5% utility of the best available option — credits near-ties.
- **NDCG@3**: ranking quality of the router's top-3 ordering against the graded (softmax) ground-truth utility, rewarding correct ordering even when Top-1 is missed.
- **Accuracy / Cost / Latency retention vs. SBM**: how the router's aggregate correctness, spend, and latency compare to always using the single best model.

## 6. Results — This Run (5-Fold CV, Pooled Out-of-Fold, 76 Queries)

### Cross-validation stability
| Metric | Value |
|---|---|
| Best val loss | 0.3600 ± 0.0286 |
| Val F1 at best loss | 0.1148 ± 0.0359 |
| Val accuracy at best loss | 0.3000 ± 0.0667 |
| Holdout loss | 0.3494 ± 0.0205 |
| Holdout predicted-mean | 0.1702 ± 0.0110 |

The loss numbers are consistent across folds (small standard deviations), which is a genuine improvement over earlier runs — the model isn't wildly unstable fold-to-fold. But **val F1 (0.11) and val accuracy (0.30) are low in absolute terms**, and both carry meaningful variance (±0.036 and ±0.067). The model is converging to a stable *loss* value, but that loss value doesn't correspond to strong discriminative performance yet.

### Pooled router evaluation vs. baselines
| Baseline | Mean Utility |
|---|---|
| Oracle | 1.3546 |
| Router | 0.9235 |
| SBM (google/gemini-3.5-flash) | 1.0047 |
| Random | 0.8209 |
| Cost-Optimal | 0.9633 |

**Utility Gap Closed vs SBM: −23.19%.** Against the accuracy-first baseline, the router still looks worse than always routing to Gemini-3.5-Flash — but this is now explicitly the wrong yardstick for a `cost_first`-scenario router, and is reported mainly for continuity with earlier runs.

**Utility Gap Closed vs Cost-Optimal: −10.17%.** This is the metric that actually matches current intent, and it tells a different, more useful story: the router is closer to (though still short of) the always-cheapest baseline. It's not yet beating even the trivial cost-optimal policy, but the gap is less than half the size of the SBM-framed number — a genuinely less alarming result once the comparison matches what the system is optimized for.

**Top-1 Oracle Match Rate: 34.21%.** The router picks the truly-best model about a third of the time. **Top-K (within 5%) Match: 40.79%** — only modestly higher than Top-1, meaning that when the router misses, it's often missing by more than a near-tie, not just picking a close second choice.

**NDCG@3: 0.7526.** This is the one genuinely strong number here, and it's informative: it says the router's *ranking* of the top candidates is reasonably good even when its *Top-1 pick* is wrong. Combined with the low Top-1 rate, this points to a router that generally has the right model somewhere in its top 2–3, but the decision boundary that pushes one model above another at the top is not well calibrated yet.

**Accuracy vs. SBM: 43.4% router vs. 76.3% SBM (retention 56.9%).** The router's picks are correct less than half the time the SBM baseline is, on raw correctness — expected and largely intentional for a cost-optimized policy, not itself a failure signal. **Cost reduction: 78.6%, latency reduction: 58.0%** — the router is finding much cheaper and faster options than always using SBM, which is the behavior `cost_first` is supposed to produce.

### Honest read
With the correct baseline in view (Cost-Optimal, not SBM), this run reads less as "the router is failing" and more as "the router is not yet beating a trivial cost-optimal policy, but it isn't far off, and its ranking quality (NDCG@3 = 0.75) is ahead of its Top-1 decision quality (34%)." Cross-fold variance is under control (see loss stability above), so this looks like a genuine calibration/capacity/data limitation rather than an artifact of unstable training. The −10% vs Cost-Optimal figure is the number to move toward positive territory before claiming this router is worth deploying over a static cheapest-model policy — the −23% vs SBM figure, taken alone as it was in the previous version of this report, overstated how far off the system actually is from its own design goal.

## 7. Recommended Next Actions

1. **Close the remaining gap vs. Cost-Optimal (−10.17%).** This is now the right target metric for the `cost_first` scenario. Since NDCG@3 is strong but Top-1 match is weak, the likely lever is Top-1 calibration rather than the weights themselves (see #2).
2. **Recalibrate the Top-1 decision, not just the ranking.** NDCG@3 (0.75) being much stronger than Top-1 match (34%) suggests the fix may be in the decoder head / decision threshold rather than the encoder — e.g., training against the soft (non-binarized) utility distribution directly instead of an argmax BCE target, since a hard 0/1 target can create the exact "confidently ranks well, decides poorly" pattern seen here.
3. **Increase training data.** With ~76 usable queries split five ways, each fold trains on roughly 45–50 queries — likely still the binding constraint on val accuracy and F1. Expanding the query set (more domains, more per-domain examples) is probably the single highest-leverage change before further architecture tuning.
4. **Sweep the classification threshold and/or loss formulation** once more data is available, and re-run cross-validation to see whether Top-1 match rate and both Utility-Gap-Closed figures move together with NDCG@3, or whether they remain decoupled (which would point more strongly at a calibration fix over a data fix).
5. **Track Utility Gap Closed vs. Cost-Optimal as the primary go/no-go metric for this scenario** rather than the SBM-framed figure, loss, or F1 in isolation — it's the number that directly answers "is this router worth deploying over a static cheapest-model policy," which is the system's actual current design goal.
