# GraphRouterV2

A graph neural network (GNN) that routes each incoming query to the best of several candidate LLMs, trading off correctness, cost, latency, and reliability. Queries and models are nodes in a bipartite graph; query→model edges carry execution features (cost, latency, correctness, tokens, completion status), and the network learns to score each edge so the highest-scoring model can be picked at inference time.

Full technical write-up (architecture, evaluation protocol, and current results): [`GraphRouterV2_Technical_Report.md`](./GraphRouterV2_Technical_Report.md).

## Candidate models

GPT-5 · Claude-Sonnet-4.5 · Gemini-3.5-Flash · DeepSeek-R1 · Qwen3-235B-Thinking · Llama-3.1-70B · Mixtral-8x22B

## Project structure

```
GraphRouterV2/
├── run_exp.py                     # Entry point: train (with k-fold CV)
├── inference.py                   # Entry point: single-query inference
├── configs/
│   └── config.yaml                # All hyperparameters, paths, utility-weight scenarios
├── model/
│   ├── multi_task_graph_router.py # Orchestration: data loading/join, splitting, training, inference
│   ├── graph_nn.py                # EncoderDecoderNet (GNN), train_validate, run_kfold_cv
│   └── router_evaluation.py       # Relative Utility Gap Closed + supporting metrics
├── data_processing/
│   ├── feature_builder.py         # Node/edge feature construction, utility label computation
│   └── utils.py
├── datasets/
│   ├── construct_router_data.py   # Builds the router benchmark via the OpenRouter API
│   ├── open_router.py
│   └── 200Q-Final/
│       ├── router_training_data.csv       # Per (query, model) execution records
│       ├── Router_Benchmark_reduced.csv   # Per-question static metadata
│       ├── LLM_Descriptions.json
│       └── llm_description_embedding.pkl
├── checkpoints/                   # Saved model weights + training dashboards, per run/fold
├── infer_results/                 # Per-query inference outputs (ranking HTML/plots)
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

Trains with 5-fold cross-validation (`kfold_cv: True` in config), saves per-fold checkpoints and dashboards under `checkpoints/`, and prints a pooled out-of-fold evaluation report (see [Evaluation](#evaluation) below) at the end of the run.

### Run inference on a single query

```bash
python inference.py --config_file configs/config.yaml --id <query_row_id>
```

Loads the best saved checkpoint, builds the graph for that query against all candidate models, and writes a ranking (predicted vs. ground-truth top-3, with scores) to `infer_results/`.

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
| `exclude_llms` | Optionally drop a model from training/eval entirely (e.g. one with incomplete data coverage) |

Utility-weight terms whose underlying data column has no variance in the current dataset are automatically zeroed out at load time (with a printed warning), so a weight can never silently be a no-op without visibility into why.

## Evaluation

Routing quality is reported via `router_evaluation.py`, headlined by **Relative Utility Gap Closed**:

```
(U_router − U_baseline) / (U_oracle − U_baseline) × 100
```

reported against two baselines — Single-Best-Model (accuracy-first) and Cost-Optimal (always cheapest) — since which one is the right comparison depends on the active `utility_scenario`. Supporting metrics: Top-1 Oracle Match Rate, Top-K-within-tolerance Match, NDCG@3 (ranking quality), and accuracy/cost/latency retention vs. SBM.

Evaluation runs on **pooled out-of-fold predictions** across all k folds rather than averaging per-fold scores, for a lower-variance estimate on a small dataset.

See the [technical report](./GraphRouterV2_Technical_Report.md) for the current model's results and open issues.
