# GraphRouterV2

A graph neural network (GNN) that routes each incoming query to the best of several candidate LLMs, trading off correctness, cost, latency, and reliability. The graph has three node types — Question, Reasoning (shared domain-macro-category prototypes), and Model — with query→model edges carrying execution features (cost, latency, correctness, tokens, completion status), and the network learns to score each edge so the highest-scoring model can be picked at inference time.

Full technical write-up (architecture, evaluation protocol, ablation study, and current results): [`GraphRouterV2_Technical_Report.md`](./GraphRouterV2_Technical_Report.md).

## Candidate models

GPT-5 · Claude-Sonnet-4.5 · Gemini-3.5-Flash · DeepSeek-R1 · Qwen3-235B-Thinking · Llama-3.1-70B · Mixtral-8x22B

## Project structure

```
GraphRouterV2/
├── run_exp.py                        # Entry point: train (with k-fold CV)
├── inference.py                      # Entry point: single-query inference
├── test_inductive_generalization.py  # Tests a held-out (never-trained-on) LLM, no retraining
├── configs/
│   └── config.yaml                   # Hyperparameters, paths, utility-weight scenarios, ablation flags
├── model/
│   ├── multi_task_graph_router.py    # Orchestration: data loading/join, splitting, training, inference,
│   │                                  #   explainability calls, checkpoint resolution
│   ├── graph_layers.py               # FeatureAlign + EncoderDecoderNet (the GNN), checkpoint/experiment naming
│   ├── gnn_trainer.py                # GNN_prediction: train/validate/test loop, ablation-results CSV logging
│   ├── kfold_cv.py                   # Leakage-free k-fold CV (fits fresh encoders per fold)
│   ├── gnn_kfold_legacy.py           # Old leaky k-fold CV, kept for reference only — not on the active path
│   ├── graph_data.py                 # Builds the PyTorch Geometric Data object: 3 node types (Question,
│   │                                  #   Reasoning prototype, Model) with a dedicated Question<->Reasoning
│   │                                  #   structural edge set; see technical report §5 for verification status
│   ├── router_utility.py             # Utility-score formula + ranking metrics (Spearman, NDCG@3, top-1)
│   ├── explainability.py             # Rule-based explain_selection() — "why this model was picked"
│   ├── data_integrity.py             # Rectangular query/model blocks, ordering, train/val/test splits
│   ├── router_plotting.py            # Predicted-scores / ground-truth ranking charts
│   ├── gnn_plotting.py               # Training-diagnostic plots (loss curves, F1/accuracy, dashboard)
│   └── io_utils.py                   # JSON/pickle load & save helpers
├── data_processing/
│   ├── feature_builder.py            # Node/edge feature construction, utility label computation
│   └── utils.py
├── datasets/
│   ├── construct_router_data.py      # Builds the router benchmark via the OpenRouter API
│   ├── open_router.py
│   └── 200Q-Final/
│       ├── router_training_data.csv       # Per (query, model) execution records
│       ├── Router_Benchmark_reduced.csv   # Per-question static metadata
│       ├── LLM_Descriptions.json
│       └── llm_description_embedding.pkl
├── checkpoints/                      # Saved model weights + dashboards, per run/fold/ablation-arm
│   └── ablation_results.csv          # One row per training run, across all ablation arms (default path)
├── infer_results/                    # Per-query inference outputs (ranking HTML/plots + explanation)
└── GraphRouterV2_Technical_Report.md
```

## Setup

```bash
pip install -r requirements.txt   # torch, torch_geometric, pandas, numpy, pyyaml, python-dotenv, wandb (optional)
```

Create a `.env` file (used by `datasets/construct_router_data.py` and `run_exp.py`) with any required API keys, e.g.:

```
OPENROUTER_API_KEY=...
```

## Usage

### Train

```bash
python run_exp.py --config_file configs/config.yaml
```

Trains with 5-fold cross-validation (`kfold_cv: True`), fitting encoders/scalers fresh inside each fold (no cross-fold leakage — see the technical report §8). Saves per-fold checkpoints and dashboards under `checkpoints/` (nested under an ablation-arm subfolder if any ablation flag is non-default — see [Ablation Study](#ablation-study) below), and prints a pooled out-of-fold evaluation report at the end of the run.

### Run inference on a single query

```bash
python inference.py --config_file configs/config.yaml --id <query_row_id>
```

Loads the best saved checkpoint for the active config (resolved the same way training resolved it, so they always agree), builds the graph for that query against all candidate models, writes a ranking (predicted vs. ground-truth top-3, with scores) to `infer_results/`, and now also prints a short rule-based explanation of the winning pick (see [Explainability](#explainability) below).

### Test inductive generalization

```bash
python test_inductive_generalization.py --config_file configs/config.yaml
```

Trains (or reuses) a checkpoint with one LLM fully excluded, then splices that excluded model back in at inference time — using its real data and the already-fitted encoders, no retraining — and reports how its predicted score/rank compares to its ground-truth utility/rank, flagged against a heuristic check for whether the prediction looks like a genuine ranking versus an uninformed guess.

## Configuration

Key fields in `configs/config.yaml`:

| Field | Purpose |
|---|---|
| `data_dir`, `benchmark_path` | Paths to router training data and benchmark metadata |
| `llm_num` | Number of candidate models |
| `train_epoch`, `learning_rate`, `weight_decay`, `dropout` | Training hyperparameters |
| `kfold_cv`, `kfold_k`, `kfold_inner_val_ratio` | Cross-validation settings |
| `early_stopping_patience` | Epochs without val-loss improvement before stopping |
| `utility_scenario` | Which named utility-weight profile is active: `performance_first`, `balance`, or `cost_first` |
| `utility_weight_profiles` | The weight sets themselves (`w_success`, `w_cost`, `w_latency`, `w_output_tokens`, `w_completion_reliability`) |
| `exclude_llms` | Optionally drop a model from training/eval entirely (also used by the inductive-generalization test) |
| `ablation_disable_node_metadata` | Drops node metadata from the Question node's feature vector |
| `ablation_edge_aware` | Set `false` to disable edge-feature-aware message passing in `GeneralConv` |
| `ablation_results_csv` | Where each training run's summary row is appended (default `checkpoints/ablation_results.csv`) |

Utility-weight terms whose underlying data column has no variance in the current dataset are automatically zeroed out at load time (with a printed warning), so a weight can never silently be a no-op without visibility into why.

## Explainability

`explain_selection()` produces a short, rule-based explanation for each routing decision. Two tiers: when historical outcome data for the query is available, it ranks each active criterion (correctness, reliability, cost, latency, output length) by its weighted-contribution gap vs. the runner-up model; for true blind inference on a new query, it falls back to the predicted-score margin alone. Always up to three reasons, best-first, e.g. "Selected Model: X / Reasons: lowest cost among correct candidates; ...". This is an auditable rule layer over existing features and utility weights, not a learned/attention-based explanation.

## Ablation Study

Two independent toggles in `config.yaml` (§Configuration above) let you disable node metadata and/or edge-aware message passing without touching code. Checkpoints for each flag combination land in their own subfolder automatically:

```
<model_path>/                             # both defaults
<model_path>/no_metadata/                 # metadata ablation only
<model_path>/no_edge_aware/               # edge-aware ablation only
<model_path>/no_metadata_no_edge_aware/   # both
```

Every training run appends one row to `ablation_results_csv`, labeled by semantic-embedding flag + both ablation flags + `utility_scenario`, with best val F1/loss, final test predict/golden, epochs run, and whether the run stopped early — so all arms accumulate into one comparable table over time. See the technical report for current coverage: 6 of the original 7 arms are runnable via existing config toggles today; the Reasoning-Node arm now has its graph-construction side implemented (`graph_data.py`) but isn't yet wired into a toggle or independently verified end-to-end, and only one arm's results have been collected so far.

## Evaluation

Routing quality is reported via the pooled out-of-fold evaluation suite, headlined by **Relative Utility Gap Closed**:

```
(U_router − U_baseline) / (U_oracle − U_baseline) × 100
```

reported against two baselines — Single-Best-Model (accuracy-first) and Cost-Optimal (always cheapest) — since which one is the right comparison depends on the active `utility_scenario`. Supporting metrics: Top-1 Oracle Match Rate, Top-K-within-tolerance Match, NDCG@3 (ranking quality), and accuracy/cost/latency retention vs. SBM.

See the [technical report](./GraphRouterV2_Technical_Report.md) for the current model's results, ablation coverage, and open issues.