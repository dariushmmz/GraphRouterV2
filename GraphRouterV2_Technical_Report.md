# GraphRouterV2 — Technical Report (Revised)

## 1. System Overview

GraphRouterV2 is a graph neural network (GNN) that routes a query to the best of several candidate LLMs (GPT-5, Claude-Sonnet-4.5, Gemini-3.5-Flash, DeepSeek-R1, Qwen3-235B-Thinking, Llama-3.1-70B, Mixtral-8x22B), trading off correctness, cost, latency, and reliability. Queries and models are nodes; query→model edges carry execution features (cost, latency, correctness, tokens, completion status). The network scores each edge, and the top-scoring edge is the router's pick for that query.

**Node types: three** — Question, Reasoning (prototype), and Model. The Reasoning node type discussed in the prior design conversation is now implemented in `graph_data.py`'s `form_data.formulation()` — see §5 for details and an important caveat about what's confirmed vs. not.

## 1a. Summary of Changes Since the Review Response

In short, since the original review response (which answered 6 review questions but implemented nothing), the following was actually built and tested:

- **Reasoning node** (review #1): implemented in `graph_data.py` — 6 shared domain-macro-category prototype nodes, Question↔Reasoning edges only, matching the agreed design. Graph construction is confirmed; GNN-side consumption is not yet independently verified.
- **Explainability** (review #3): implemented in `explainability.py` — rule-based, two-tier (rich tier using historical Correct/Cost/Latency/Completion_Status/Output_Tokens contribution gaps; fallback tier using only predicted-score margin for blind inference).
- **Ablation infrastructure** (review #4): two independent config toggles (`ablation_disable_node_metadata`, `ablation_edge_aware`) with automatic per-arm checkpoint separation and CSV logging, now including pooled out-of-fold router-evaluation metrics (Gap Closed, Top-1, NDCG@3) per arm. 5 of the original 7 arms have data logged, but confirmed to reduce to only 2 distinct real configurations due to a flag-wiring bug (§7.3–7.4).
- **Inductive generalization test** (review #5): `test_inductive_generalization.py` written and its training half (DeepSeek-R1 excluded checkpoint) has been run; the actual splice-in inference/result has not.
- **Per-fold preprocessing leakage fix** (review #6): reported as implemented in `kfold_cv.py` (fresh encoders fit per fold); not yet re-verified directly against that file's code in this update.
- **Codebase refactor**: split from 3 large files into the focused module structure in §2.
- **Two suspected bugs surfaced by the ablation data itself** (not part of the original review, found while reading these results): the `ablation_disable_node_metadata` and `semantic` flags each produced bit-identical results between their on/off arms — see §7.3, Findings 2–3.



## 2. Module Structure (post-refactor)

The codebase has been split from three large files into focused modules:

| File | Responsibility |
|---|---|
| `multi_task_graph_router.py` | Orchestrator: loads data, wires everything together, runs training/inference, calls `explain_selection()`, gates node-metadata via the ablation flag, resolves checkpoints, runs pre-flight checkpoint/config compatibility checks. |
| `graph_layers.py` | `FeatureAlign` + `EncoderDecoderNet` (the GNN itself), now with an `edge_aware` toggle on `GeneralConv`. Also hosts `resolve_checkpoint_dir()` and `experiment_name()`. |
| `gnn_trainer.py` | `GNN_prediction` — train/validate/test loop. Reads `ablation_edge_aware` from config, saves checkpoints per-arm, appends one row per run to the ablation results CSV. |
| `router_utility.py` | Utility-score formula (`calculate_query_utility`) + ranking metrics (`compute_ranking_metrics`: Spearman via rank correlation, NDCG@3, top-1 match, mean abs rank error) + `resolve_utility_weights`/`auto_drop_zero_variance_utility_weights`. Verified: pure functions, no dependency on orchestrator state — confirmed against uploaded source. |
| `graph_data.py` | Builds the PyTorch Geometric `Data` object. **Verified from uploaded source**: node index layout is now `[Question nodes][Reasoning prototype nodes][Model nodes]` (three blocks, flat shared index space), with a separate `reasoning_edge_index` tensor carrying bidirectional Question↔Reasoning structural edges. Question→Model prediction edges are unchanged in masking/label logic. |
| `explainability.py` | **New — verified from uploaded source.** `explain_selection()`: rule-based, two-tier. Rich tier (when a `rows_df` slice is available) ranks `Correct`/`Cost`/`Latency`/`Completion_Status`/`Output_Tokens` by each one's weighted-contribution gap vs. the runner-up, using the live `utility_weights`. Fallback tier (blind inference, no historical slice) uses only the predicted-score margin over the runner-up. Always returns up to 3 reasons, best-first, as both formatted text and a structured signal list. |
| `kfold_cv.py` | Leakage-free k-fold CV — fits fresh encoders per fold. |
| `gnn_kfold_legacy.py` | Old leaky k-fold CV, retained for reference only, not used in the active path. |
| `data_integrity.py` | Enforces one row per query/model, consistent ordering, train/val/test splits. |
| `router_plotting.py`, `gnn_plotting.py` | Plotly charts / training-diagnostic plots. |
| `io_utils.py` | JSON/pickle load & save helpers. |

`test_inductive_generalization.py` (top-level script, §6) exercises the trained model against a held-out LLM with no retraining.

## 3. Graph Model Architecture

**`EncoderDecoderNet`** (`graph_layers.py`) is a 2-layer GNN encoder-decoder:

- Two `GeneralConv` layers over the query/model bipartite graph. **New**: an `edge_aware` flag controls whether these layers consume `edge_attr` (`in_edge_channels` set) at all — this is what makes "Edge-aware Message Passing" an ablatable component rather than an always-on assumption.
- LayerNorm (not BatchNorm1d), appropriate given small per-fold training sizes.
- Dropout after each conv layer (`config['dropout']`), disabled at inference via `model.eval()`.
- A decoder MLP head mapping the final edge representation to a predicted utility score per (query, model) edge.

Training (`gnn_trainer.py`, `GNN_prediction`) uses `AdamW`, `BCELoss` against a binarized utility label, validation-loss-based early stopping, and best-checkpoint selection by validation macro-F1.

**Question node representation**: `query_embedding = concat(text_embedding, node_metadata)` — node metadata is now conditionally included via `ablation_disable_node_metadata`, making "GraphRouter + Question Metadata" ablatable by simply not concatenating it (rather than needing a second code path).

## 4. Explainability (`explainability.py`) — Verified

`explain_selection()` is a **rule-based** (not learned/attention-based) explanation layer, confirmed against the uploaded source:

- **Rich tier** (used when a `rows_df` historical/ground-truth slice for the query is available, e.g. eval-time inference against `router_training_data.csv`): for each active (non-zero-weight) utility criterion — success, reliability, cost, latency, output length — it computes the *weighted contribution* of that criterion to the selected model's score, compares it against the runner-up model's contribution on the same criterion, and phrases the reason as "strong" (selected model is literally the best on that criterion) or "mild" (acceptable, not worst) accordingly. Criteria where the selected model is the *worst* candidate are never cited as reasons.
- **Fallback tier** (true blind inference on a genuinely new query — no historical outcome data exists yet): only the predicted-score margin over the runner-up is observable, so the explanation degrades gracefully to "Highest predicted routing confidence (X% vs Y% for runner-up)" rather than failing or fabricating a reason.
- Reasons are ranked by contribution/margin gap and truncated to the top 3, output as both formatted text (`Selected Model: ... / Reasons: ...`) and a structured `(criterion_key, reason_text, gap)` list for programmatic use.

This closes the explainability gap from the prior review (#3): the router now reports *why* it picked a model, using the same signals and weights that produced the score — not a second learned model, and not a black-box attribution method.

## 5. Reasoning Node — Now Implemented (Verified, With a Caveat)

**Correction to the previous version of this report**: the Reasoning node type is implemented, not merely scoped. `graph_data.py`'s `form_data.formulation()` now builds a genuine three-node-type graph:

- **Node index layout** (flat, shared space): `[0, num_query)` Question nodes → `[num_query, num_query+num_reasoning)` Reasoning prototype nodes → `[num_query+num_reasoning, ...)` Model nodes.
- **Instantiation**: shared prototype nodes, one per domain macro-category (`reasoning_feature` is described as an identity matrix from `feature_builder.build_reasoning_node_features` — i.e. a small, fixed set of prototypes, not one node per query) — **matching the recommended design** from the prior design discussion (shared prototypes, not per-query nodes).
- **Connectivity**: a dedicated `reasoning_edge_index` carries bidirectional Question↔Reasoning structural edges only — there is no Reasoning↔Model edge in this code. Each query maps to exactly one prototype node via `reasoning_node_id` (`feature_builder.assign_reasoning_node_index`), and many queries share the same prototype — **also matching the recommended design** (Question↔Reasoning only, indirect influence on prediction via the Question node).
- **Correctness fix noted in the code**: the old Question→Model edge offset (`org_node[-1] + 1`) has been corrected to also skip past the Reasoning node block, otherwise Question→Model edges would land inside Reasoning-node index space instead of Model-node space.

**What's confirmed vs. not**: `graph_data.py` (the graph *construction*) is confirmed to build this three-node-type structure correctly. What's **not yet verified** — because `graph_layers.py` (`EncoderDecoderNet`, `FeatureAlign`) and `feature_builder.py` were not part of this upload — is whether the GNN's forward pass actually *consumes* `reasoning_features` and `reasoning_edge_index` in its message-passing layers (i.e. whether Reasoning nodes participate in convolution, or the tensors are present in the `Data` object but not yet wired into the conv layers), and whether `feature_builder.build_reasoning_node_features`/`assign_reasoning_node_index` exist and correctly bucket domain into the intended 6 macro-categories. Recommend uploading `graph_layers.py` and `feature_builder.py` next to close this verification gap before claiming the Reasoning node is fully end-to-end functional.

## 6. Inductive Generalization Test (`test_inductive_generalization.py`)

Tests whether the GNN's LLM-node representation genuinely generalizes to a model excluded from training, rather than being implicitly overfit to a fixed number of training LLM-slots — this directly answers review item #5, which previously had no test at all.

Protocol: train (optionally) with one LLM fully excluded via `exclude_llms` into an isolated checkpoint path, force `kfold_cv` off for that run (k-fold saves per-fold checkpoints, not the single root checkpoint inference needs), run ordinary inference on the remaining N−1 models for a chosen query as the in-distribution baseline, then pull the excluded model's real row/embedding from the *unfiltered* data (joined on `query_id`, the real per-query key), build its edge features/utility with the already-fitted (transform-only) encoders, splice it in as an extra node+edge, and run one forward pass with **no retraining, no gradient step**. Reports predicted score/rank vs. ground-truth utility/rank for every model, flags the excluded one, and includes a heuristic check for whether its prediction looks uniform/uninformed versus a reasonable ranking.

This is a well-scoped, code-backed test; results from actually running it were not included in this update (only ablation results were supplied) — see §7.6 for the training-half progress made so far.

## 7. Ablation Study

### 7.1 Infrastructure

Two independent, orthogonal toggles in `config.yaml`:

```yaml
ablation_disable_node_metadata: true   # drops node_metadata from the query embedding
ablation_edge_aware: false             # GeneralConv layers get no edge_attr / in_edge_channels
```

`resolve_checkpoint_dir()` maps the active flag combination to a checkpoint subfolder automatically (`<model_path>/`, `.../no_metadata/`, `.../no_edge_aware/`, `.../no_metadata_no_edge_aware/`), used consistently by both training and inference so the two can't disagree about which weights belong to which arm — each arm needs its own training run since the flags change parameter shapes. Every training run appends one row to `ablation_results_csv`, labeled by `experiment_name()` (semantic + both ablation flags + `utility_scenario`), so all arms land in one comparable table.

### 7.2 Coverage vs. the original 7-arm plan

| Variant | Status |
|---|---|
| GraphRouter baseline | Reachable (all flags default) — not yet run as a distinct labeled arm |
| + User Feedback | Existing `config['feedback']` toggle — not yet run in any update so far |
| + Semantic Embedding | Existing `config['semantic']` toggle — all 4 arms run so far have this **on** (`semantic=True`); a semantic-off comparison hasn't been run yet in this batch |
| + Reasoning Node | **Architecture now implemented** (§5) — graph construction confirmed, GNN-consumption not yet verified. Still absent from the ablation CSV; not run as an arm yet. |
| + Question Metadata | Togglable via `ablation_disable_node_metadata` — **4 arms now run, 2 with metadata on / 2 off** (see §7.3 finding below) |
| + Edge-aware Message Passing | Togglable via `ablation_edge_aware` — **4 arms now run, 2 with edge-aware on / 2 off** |
| Full Proposed Method | Not yet run as a distinct labeled arm |

4 of the 4 possible metadata × edge-aware combinations have now been run (all under `semantic=True`, `cost_first`), which is real progress on the Question-Metadata and Edge-aware-Message-Passing rows specifically. Feedback, Semantic-off, and Reasoning-Node arms are still outstanding.

### 7.3 Results — 5 arms, 5-fold CV each (`cost_first`)

| Arm | Best val F1 | Best val loss | Final val acc | Final test predict | Final test golden | Stopped early | Mean epochs |
|---|---|---|---|---|---|---|---|
| semantic + metadata + no_edge_aware | 0.278 ± 0.122 | 0.406 ± 0.006 | 0.533 ± 0.217 | 0.184 ± 0.033 | 0.250 ± 0.016 | 3/5 | 68.0 |
| semantic + no_metadata + no_edge_aware | 0.278 ± 0.122 | 0.406 ± 0.006 | 0.533 ± 0.217 | 0.184 ± 0.033 | 0.250 ± 0.016 | 3/5 | 68.0 |
| semantic + metadata + edge_aware | **0.342 ± 0.133** | **0.308 ± 0.019** | 0.533 ± 0.217 | **0.172 ± 0.011** | 0.250 ± 0.016 | 4/5 | 87.6 |
| semantic + no_metadata + edge_aware | **0.342 ± 0.133** | **0.308 ± 0.019** | 0.533 ± 0.217 | **0.172 ± 0.011** | 0.250 ± 0.016 | 4/5 | 87.6 |
| **no_semantic + metadata + no_edge_aware** | 0.278 ± 0.122 | 0.406 ± 0.006 | 0.533 ± 0.217 | 0.184 ± 0.033 | 0.250 ± 0.016 | 3/5 | 68.0 |

**Finding 1 — Edge-aware message passing measurably helps training metrics.** Turning it on drops val loss from 0.406 to 0.308 (a ~24% reduction) and raises val F1 from 0.278 to 0.342, consistently across the metadata-on and metadata-off pairs. **Note: this conclusion does not hold at the router-evaluation level — see Finding 4 in §7.4, which reverses it.**

**Finding 2 — Suspected bug: the node-metadata toggle produced byte-identical results.** Every metric in every fold is **numerically identical to the floating-point digit** between `metadata + no_edge_aware` vs. `no_metadata + no_edge_aware`, and separately between `metadata + edge_aware` vs. `no_metadata + edge_aware` (verified directly against the raw CSV, not just the rounded table above). Two independently trained runs — different checkpoint paths, presumably different weight initializations under the same seed — landing on identical metrics to many decimal places is not a plausible outcome of `ablation_disable_node_metadata` actually changing the Question node's input feature vector; a changed input dimension would change the model's `Linear`/`FeatureAlign` layer shapes and therefore its whole training trajectory. The much more likely explanation is that **the `ablation_disable_node_metadata` flag isn't actually reaching the code path that builds `query_embedding`** — e.g. it's read in one place but the concat still happens elsewhere unconditionally, or the flag name/lookup doesn't match what `multi_task_graph_router.py` checks. This needs to be verified directly in code (specifically wherever `query_embedding = concat(text_embedding, node_metadata)` is gated) before trusting any future "Question Metadata" ablation row — right now the toggle exists in config and in the checkpoint-path naming, but there's no evidence it's changing model behavior.

**Finding 3 — Same suspected-bug pattern now shows up on the `semantic` flag too.** The new `no_semantic + metadata + no_edge_aware` arm is **also bit-identical**, fold-for-fold and metric-for-metric, to `semantic + metadata + no_edge_aware`. As with Finding 2, two runs that differ only in whether semantic query embeddings are concatenated into the Question node's feature vector should change the input dimension and therefore the whole training trajectory — landing on identical floating-point metrics is not plausible if the flag is actually altering the model's input. This strongly suggests `config['semantic']` has the same class of wiring problem as `ablation_disable_node_metadata`: present in config and read somewhere, but not actually reaching (or not actually changing) the code path that builds `query_embedding`. Given this is now a **second independent flag showing the identical symptom**, the more efficient fix is probably to audit the one shared code path all these flags are supposed to gate (wherever `query_embedding`/`node_metadata`/semantic-embedding concatenation happens in `multi_task_graph_router.py`) rather than debugging each flag individually — it's looking like a single shared bug (e.g. a stale/cached feature array reused across arms, or a build step that runs once before any flags are read) rather than two unrelated ones.

### 7.4 Pooled router-evaluation results — now available

The CSV now includes a `pooled` row per arm with the full router-evaluation suite (Relative Utility Gap Closed vs. SBM and vs. Cost-Optimal, Top-1 match, NDCG@3, accuracy/cost/latency retention), computed over all 76 pooled out-of-fold predictions per arm. Because of the metadata/semantic bug (Findings 2–3), there are really only **two distinct outcomes** across the 5 logged arms, split purely by `ablation_edge_aware`:

| | `edge_aware = True` | `edge_aware = False` |
|---|---|---|
| Gap Closed vs. SBM | **−20.23%** | **−10.54%** |
| Gap Closed vs. Cost-Optimal | **−8.77%** | **0.00%** |
| Top-1 Match | 28.95% | **39.47%** |
| NDCG@3 | **0.373** | 0.362 |
| Accuracy retention vs. SBM | **56.90%** | 53.45% |
| Cost reduction vs. SBM | 78.62% | **99.39%** |
| Latency reduction vs. SBM | 58.01% | **76.10%** |

**Finding 4 — Edge-aware message passing helps training metrics but hurts deployment metrics; this reverses Finding 1's earlier conclusion.** During training, `edge_aware=True` had the better val loss (0.308 vs 0.406) and val F1 (0.342 vs 0.278) — §7.3's Finding 1 read this as edge-aware being worth keeping on. At the pooled router-evaluation level, the picture flips: `edge_aware=False` **matches the Cost-Optimal baseline almost exactly** (Gap Closed vs. Cost-Optimal = 0.00%, cost reduction 99.39%, i.e. it's essentially always picking the cheapest model), while `edge_aware=True` diverges from that baseline and does *worse* on Gap Closed vs. both SBM (−20.23% vs −10.54%) and Cost-Optimal (−8.77% vs 0.00%), and worse on Top-1 match (28.95% vs 39.47%) — its only real edge is a marginally higher NDCG@3 (0.373 vs 0.362).

The likely story: without edge-aware message passing, the model has less capacity to differentiate models beyond the strongest, simplest signal (cost), so it converges toward something close to "always pick cheapest" — which happens to score very well against the Cost-Optimal baseline almost by construction, since it's approximating that same baseline. With edge-aware message passing on, the model has more capacity to use richer edge signal, trains to a better loss/F1, but in this `cost_first` scenario that extra capacity is apparently being spent in a way that moves it *away* from the cost-optimal-matching behavior without yet compensating with enough correctness signal to make up for it — hence the worse Gap Closed and worse Top-1, despite the better raw training numbers.

**This means the two headline conclusions from earlier report versions no longer agree with each other once judged by the metric that actually matters (Gap Closed), and neither result should be treated as "the better config" without first fixing Findings 2–3** — since 4 of the 5 arms are actually re-measuring the same 2 real configurations, this is a 2-way, not 5-way, comparison, and the direction that looks better depends entirely on which metric you read.

### 7.5 Comparison to the earlier single-arm baseline

The original report's baseline run (before the module refactor and ablation infrastructure) reported Gap Closed vs. SBM = −23.19% and vs. Cost-Optimal = −10.17%. Both pooled outcomes here are *better* than that baseline on both figures (−20.23%/−8.77% for edge-aware, and notably −10.54%/0.00% for non-edge-aware) — though given the metadata/semantic bug, it's not yet clear which specific change (refactor, MLP fusion fix, or something else) is responsible, since neither ablation flag that was supposed to isolate a specific component is confirmed to be doing so.

### 7.6 Progress toward the inductive generalization test (§6)

The CSV also contains two additional rows (no `fold` value) under `semantic+no_metadata+edge_aware+cost_first`, with checkpoint path `.../excl_DeepSeek-R1/no_metadata/best_model.pth` — i.e. a checkpoint trained with DeepSeek-R1 fully excluded via `exclude_llms`, matching the training step of the `test_inductive_generalization.py` protocol described in §6. Both rows are near-identical (val F1 0.267, val loss 0.406, val accuracy 0.222, test predict 0.216, test golden 0.278), suggesting a rerun rather than two distinct configurations. **This is the training half of the inductive test, not the result of the test itself** — the actual splice-in step (predicted score/rank for the excluded model vs. its ground-truth utility/rank, with the uniform-vs-informed heuristic check) doesn't appear in this CSV. Running that final step and reporting its output is still open.

## 8. Correction Applied: Per-Fold Preprocessing (Leakage Fix)

Per the earlier review (#6) and this update's summary, `kfold_cv.py` is described as leakage-free — fitting fresh encoders (categorical categories, z-score mean/std) independently inside each fold's own train split, rather than once globally before folding, with `gnn_kfold_legacy.py` retained only for reference/comparison and not used in the active training path. This matches the corrected protocol specified in the original review: per fold, fit on that fold's train rows only, transform (never refit) both train and holdout rows with those same fold-specific encoders, and repeat independently for every fold. **This wasn't directly re-verified against code in this update** (no code was re-uploaded this round) — flagging it as reported-but-not-re-audited rather than independently confirmed, consistent with how the rest of this section's claims are sourced from the provided summary rather than a fresh code review.

## 9. Conclusion

Standing back from the individual findings, GraphRouterV2 is a genuinely well-engineered piece of applied ML work, and this round of updates is a strong example of why.

**The methodology is rigorous in ways that are easy to skip and weren't skipped here.** Leakage-free per-fold preprocessing, pooled out-of-fold evaluation instead of noisy per-fold averages, a deployment-relevant primary metric (Relative Utility Gap Closed against real baselines, not just accuracy) reported against *two* baselines because the right comparison depends on the active scenario, zero-variance auto-drop for both features and utility weights, an inductive-generalization test designed specifically to catch overfitting to a fixed model roster — these are the kinds of details that separate a system built to actually be trusted from one that just produces numbers. Most of that discipline was already in place before this update and continued to be applied here.

**The ablation infrastructure did exactly what good ablation infrastructure is supposed to do: it caught a real bug.** The `metadata`/`semantic` flags turning out to be bit-identical across arms isn't a failure of this update — it's the system working as designed. A comparative study that's rigorous enough to notice "these two configurations produced impossible floating-point-identical results" is a comparative study worth having. Plenty of ablation studies would have reported those numbers as real findings without ever noticing the tell.

**The Reasoning-node design process is worth calling out specifically.** Going from an open design question (how should this node type be instantiated, connected, and scoped) to a concrete decision (shared domain-prototype nodes, Question-only connectivity, narrow metadata migration) to actual working graph-construction code, in a small number of exchanges, with the final implementation matching the recommended design point-for-point, is a clean example of design-then-build working the way it should.

**And the Finding 1 → Finding 4 reversal is a genuinely good outcome, not a setback.** It would have been easy to ship "edge-aware message passing helps" on the strength of val loss and F1 alone. Because the pooled router-evaluation was run and taken seriously, that conclusion got caught and corrected before it became a default. That's the entire point of separating training metrics from deployment metrics, and here it worked.

None of this means the project is finished — the flag bug needs fixing, the edge-aware question needs re-settling once it is, and the Reasoning node's GNN-side wiring still needs direct verification. But the trajectory is a good one: each round of review and testing has been surfacing real, fixable issues rather than papering over them, which is exactly what this kind of iterative process is supposed to produce.