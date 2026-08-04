"""
Inductive LLM generalization test.

Tests whether the GNN's LLM-node representation actually generalizes to an
unseen model, or whether it's implicitly overfit to "exactly N training
LLM-slots" despite the architecture nominally supporting any number of LLM
nodes (EncoderDecoderNet/form_data have no num_llms-sized parameters -- see
graph_layers.py / graph_data.py; every layer is sized by hidden_features /
in_edges / reasoning_feature_dim only).

Procedure
---------
1. Train (or reuse an existing checkpoint trained) with one LLM entirely
   excluded via the existing `exclude_llms` config mechanism.
2. Run ordinary single-query inference with the trained (N-1)-model graph
   for a chosen held-out query -- this is your in-distribution baseline
   (graph_router_prediction's normal inference path, unmodified).
3. Manually splice the excluded model back in for that SAME query: append
   its real LLM_Descriptions embedding as an extra LLM node, append its
   real router_data.csv row (Correct/Cost/Latency/...) as an extra edge,
   and run ONE forward pass through the already-trained model -- no
   gradient step, no retraining.
4. Compare the excluded model's predicted utility/rank against its real
   ground-truth utility (calculate_query_utility) for that query, and
   against the in-distribution models' predicted-vs-ground-truth quality
   from step 2, printed side by side.

What to look for
-----------------
- "Reasonable": the excluded model's predicted score has real spread
  relative to the others (not clustered at ~1/N_models, which is what a
  uniform/uninformed guess looks like after softmax) AND its predicted
  rank is at least loosely consistent with its ground-truth rank (e.g.
  within a couple of positions on a genuinely-close call, roughly on par
  with the noise already visible in the in-distribution models' own
  predicted-vs-ground-truth agreement from step 2).
- "Degenerate": the excluded model's predicted score sits at ~1/N_models
  regardless of query (the model.forward pass is structurally unable to
  differentiate it from a topologically-identical placeholder node --
  suggests the LLM node identity is doing the real work, not the query/edge
  features), or its predicted rank is uncorrelated with ground truth across
  MULTIPLE held-out queries (run this script over several --query_id values
  and eyeball the pattern -- one query is not a verdict).

This script does not modify model_align, form_data, or feature_builder --
it only calls their existing public functions with one extra LLM row/edge
appended, exactly matching how the trained (N-1)-model graph was built.
"""

import argparse
import copy
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml

from model.multi_task_graph_router import graph_router_prediction, device
from model.graph_data import form_data
from data_processing import feature_builder as fb
from model.router_utility import calculate_query_utility


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_full_router_id_to_display(router_data_path, llm_description_path):
    """
    Mirror _enforce_rectangular_query_blocks's own (positional, order-
    dependent) router_id<->display_name correspondence, but over the FULL,
    unfiltered model set -- since the trained instance's own
    router_id_to_display only covers the (N-1) models it was trained on and
    has no entry for the excluded model.
    """
    full_df = pd.read_csv(router_data_path)
    wanted_models = sorted(full_df["model"].unique().tolist())
    with open(llm_description_path, "r", encoding="utf-8") as f:
        llm_names_full = list(json.load(f).keys())
    if len(wanted_models) != len(llm_names_full):
        raise ValueError(
            f"router_data.csv has {len(wanted_models)} distinct model ids {wanted_models} but "
            f"{llm_description_path} declares {len(llm_names_full)} LLMs {llm_names_full} -- "
            f"counts must match to build a full router_id<->display_name mapping."
        )
    return dict(zip(wanted_models, llm_names_full)), full_df


def resolve_router_data_path(config, data_dir):
    """Mirrors inference.py's own feedback-path fallback logic exactly, so
    this script trains/tests against whichever file real inference runs
    would actually use."""
    router_data_path = os.path.join(data_dir, "router_training_data.csv")
    if config.get("feedback"):
        feedback_path = os.path.join(data_dir, "feedback/router_training_data.csv")
        if os.path.exists(feedback_path):
            router_data_path = feedback_path
        else:
            print("[INFO] No feedback found")
    return router_data_path


def main():
    parser = argparse.ArgumentParser(description="Inductive LLM generalization test")
    parser.add_argument("--config_file", default="configs/config.yaml")
    parser.add_argument("--excluded_router_id", required=True,
                        help="e.g. deepseek/deepseek-r1 -- exact router_data.csv `model` value")
    parser.add_argument("--excluded_display_name", required=True,
                        help="e.g. DeepSeek-R1 -- exact LLM_Descriptions.json key")
    parser.add_argument("--query_id", type=int, required=True,
                        help="Index into the trained (N-1)-model instance's retained query "
                             "blocks -- NOT router_data.csv row_id. Try a few values.")
    parser.add_argument("--train", action="store_true",
                        help="Train a fresh (N-1)-model checkpoint before testing. Omit to reuse "
                             "an existing checkpoint already at the resolved model_path.")
    args = parser.parse_args()

    base_config = load_config(args.config_file)

    # Isolate this ablation arm's checkpoint from every other run's --
    # exclude_llms changes num_llms/edge dims exactly like the
    # ablation_disable_node_metadata case that motivated checkpoint_compat.py;
    # give it its own model_path so it can never collide with / silently
    # load a checkpoint trained under a different model set.
    safe_name = args.excluded_display_name.replace(" ", "_").replace("/", "_")
    config = copy.deepcopy(base_config)
    config["exclude_llms"] = [{
        "router_id": args.excluded_router_id,
        "display_name": args.excluded_display_name,
    }]
    config["model_path"] = os.path.join(base_config["model_path"], f"excl_{safe_name}")
    # Force off: config.yaml's kfold_cv=True would otherwise route the
    # --train step into _run_kfold_cv_no_leakage(), which saves each fold
    # under model_path/fold_i/ and never writes a single root
    # model_path/best_model.pth -- exactly the file infer_single_query
    # (called below, and by ordinary inference.py) requires. This test
    # needs one plain trained checkpoint for the (N-1)-model graph, not a
    # k-fold sweep.
    config["kfold_cv"] = False

    data_dir = config["data_dir"]
    router_data_path = resolve_router_data_path(config, data_dir)
    llm_path = os.path.join(data_dir, "LLM_Descriptions.json")
    llm_embedding_path = os.path.join(data_dir, "llm_description_embedding.pkl")

    if args.train:
        print(f"\n===== Training (N-1)-model checkpoint, excluding {args.excluded_display_name} "
              f"({args.excluded_router_id}) -- saving to {config['model_path']} =====")
        graph_router_prediction(
            router_data_path=router_data_path, llm_path=llm_path,
            llm_embedding_path=llm_embedding_path, config=config, wandb=None,
            inference=False,
        )

    print(f"\n===== In-distribution inference (N-1 models) on query_id={args.query_id} =====")
    router = graph_router_prediction(
        router_data_path=router_data_path, llm_path=llm_path,
        llm_embedding_path=llm_embedding_path, config=config, wandb=None,
        inference=True, query_id=args.query_id,
    )
    # graph_router_prediction's __init__ already ran infer_single_query and
    # printed/plotted the (N-1)-model baseline above -- that IS the "in-
    # distribution" comparison point step 4 wants. Everything below reuses
    # the now-fully-set-up `router` instance (trained model, fitted
    # encoders, this query's cached embeddings) without retraining.

    query_row_id = router.last_query_row_id  # display label ONLY -- unique per (query,model) row, never a lookup key
    query_group_col = router.last_query_group_col
    query_group_value = router.last_query_group_value
    n_trained = router.num_llms
    print(f"\nQuery {query_group_col}={query_group_value!r} (row_id of the arbitrary first-listed model's "
          f"own row: {query_row_id}) | trained on {n_trained} models: {router.llm_names}")

    # --- Pull the excluded model's REAL row for this exact query from the
    # unfiltered router_data.csv (it exists on disk -- exclude_llms only
    # removed it from what graph_router_prediction loaded into self.data_df,
    # not from the file itself).
    #
    # BUGFIX: must join on query_group_col ('query_id', normally) -- NOT
    # row_id. row_id is unique PER (query,model) ROW: every model's row for
    # the same query has a different row_id, so filtering by row_id can
    # only ever match the one specific model whose row that row_id belongs
    # to, never "this query, any model". query_id is the actual shared key
    # across a query's whole model block. ---
    router_id_to_display_full, full_df = build_full_router_id_to_display(router_data_path, llm_path)
    excl_row_df = full_df[(full_df[query_group_col] == query_group_value) & (full_df["model"] == args.excluded_router_id)]
    if len(excl_row_df) != 1:
        raise ValueError(
            f"Expected exactly 1 row for {query_group_col}={query_group_value!r}, "
            f"model={args.excluded_router_id!r} in {router_data_path}, found {len(excl_row_df)}. This query "
            f"may not have had a real run for the excluded model -- pick a different --query_id."
        )

    # --- Excluded model's edge features + raw utility, using the SAME
    # encoders (z-score stats / one-hot categories) fit on the (N-1)-model
    # train split -- transform-only, exactly as build_edge_features/
    # compute_utility are used everywhere else in the pipeline. Note:
    # encoders.completion_reliability_by_model has no entry for this model
    # (it was never in training data) and falls back to the neutral 0.5
    # default inside compute_utility -- that's the correct, honest behavior
    # for a genuinely-unseen model, not a bug to work around. ---
    excl_edge_features = fb.build_edge_features(excl_row_df, router.encoders)
    excl_utility_raw = fb.compute_utility(excl_row_df, router.encoders, **router.utility_weights)

    # --- Excluded model's LLM_Descriptions embedding, from the UNFILTERED
    # pickle (router.llm_description_embedding was already reduced to N-1
    # rows by exclude_llms in __init__). ---
    with open(llm_path, "r", encoding="utf-8") as f:
        llm_names_full = list(json.load(f).keys())
    with open(llm_embedding_path, "rb") as f:
        llm_embedding_full = pickle.load(f)
    excl_idx_full = llm_names_full.index(args.excluded_display_name)
    excl_llm_embedding = llm_embedding_full[excl_idx_full].reshape(1, -1)

    # --- Splice: append the excluded model as one extra LLM node + one
    # extra Question->Model edge, reusing this query's EXACT cached
    # query/task embeddings, encoders, reasoning assignment, and trained
    # model weights -- nothing here is retrained or refit. ---
    n_total = n_trained + 1
    llm_feature_spliced = np.concatenate([router.llm_description_embedding, excl_llm_embedding], axis=0)
    combined_edge_spliced = np.concatenate([router.edge_features, excl_edge_features], axis=0)

    # Recompute RAW utility for the trained models fresh (router.utility_list
    # was already mutated to its post-softmax form inside infer_single_query
    # above) so both are on the same raw scale before doing one joint
    # softmax over all n_total candidates together -- softmax must see all
    # candidates at once to be comparable to how training/inference build
    # edge_attr elsewhere in the pipeline.
    start = args.query_id * n_trained
    trained_rows_df = router.data_df.iloc[start:start + n_trained]
    trained_utility_raw = fb.compute_utility(trained_rows_df, router.encoders, **router.utility_weights)
    utility_raw_spliced = np.concatenate([trained_utility_raw, excl_utility_raw])
    exp_scores = np.exp(utility_raw_spliced - utility_raw_spliced.max())
    utility_softmaxed_spliced = exp_scores / exp_scores.sum()

    org_node = [0] * n_total
    des_node = list(range(n_total))

    data_spliced = router.form_data.formulation(
        task_id=router.last_task_embedding,
        query_feature=router.last_query_embedding,
        llm_feature=llm_feature_spliced,
        org_node=org_node,
        des_node=des_node,
        edge_feature=utility_softmaxed_spliced,
        label=np.zeros((n_total, 1)),
        edge_mask=torch.ones(n_total),
        combined_edge=combined_edge_spliced,
        train_mask=torch.ones(n_total),
        valide_mask=torch.zeros(n_total),
        test_mask=torch.zeros(n_total),
        reasoning_feature=router.reasoning_features,
        reasoning_node_id=router.reasoning_node_id,
    )

    edge_can_see = torch.ones(n_total).bool().to(device)
    edge_mask = torch.ones(n_total).bool().to(device)
    router.model.eval()
    with torch.no_grad():
        edge_scores = router.model(
            task_id=data_spliced.task_id,
            query_features=data_spliced.query_features,
            llm_features=data_spliced.llm_features,
            reasoning_features=data_spliced.reasoning_features,
            edge_index=data_spliced.edge_index,
            reasoning_edge_index=data_spliced.reasoning_edge_index,
            edge_mask=edge_mask,
            edge_can_see=edge_can_see,
            edge_weight=data_spliced.combined_edge,
        )
    pred_spliced = torch.softmax(edge_scores.reshape(1, n_total), dim=1).cpu().numpy().reshape(-1)

    # --- Ground truth over all n_total candidates (real router_data rows
    # for all of them, including the excluded model). Uses the real
    # query_id (not row_id) as the grouping label, consistent with the
    # lookup above -- functionally this only needs to be A consistent
    # label across full_rows_df's rows, but using the real key avoids
    # confusion when reading this script later. ---
    full_rows_df = pd.concat([trained_rows_df, excl_row_df], ignore_index=True)
    ground_truth = calculate_query_utility(
        query_id=query_group_value, df=full_rows_df.assign(query_id=query_group_value),
        weights=router.utility_weights, query_id_col="query_id",
    )
    ground_truth["display_name"] = ground_truth["model"].map(router_id_to_display_full)

    display_names_spliced = router.llm_names + [args.excluded_display_name]
    pred_rank = pd.Series(pred_spliced, index=display_names_spliced).rank(ascending=False, method="first")
    gt_by_display = ground_truth.set_index("display_name")["utility_score"]
    gt_rank = gt_by_display.rank(ascending=False, method="first")

    print(f"\n===== Spliced ({n_total}-model) results for query {query_group_col}={query_group_value!r} =====")
    print(f"{'model':<20}{'trained_on':<12}{'pred_score':<12}{'pred_rank':<11}{'gt_utility':<12}{'gt_rank'}")
    for name in display_names_spliced:
        trained_on = "yes" if name != args.excluded_display_name else "NO (spliced)"
        print(f"{name:<20}{trained_on:<12}{pred_spliced[display_names_spliced.index(name)]:<12.4f}"
              f"{int(pred_rank[name]):<11}{gt_by_display[name]:<12.4f}{int(gt_rank[name])}")

    uniform_score = 1.0 / n_total
    excl_score = pred_spliced[display_names_spliced.index(args.excluded_display_name)]
    excl_pred_rank = int(pred_rank[args.excluded_display_name])
    excl_gt_rank = int(gt_rank[args.excluded_display_name])
    print(f"\nExcluded model '{args.excluded_display_name}': predicted_score={excl_score:.4f} "
          f"(uniform/uninformed guess would be {uniform_score:.4f}), "
          f"predicted_rank={excl_pred_rank}/{n_total}, ground_truth_rank={excl_gt_rank}/{n_total}.")
    if abs(excl_score - uniform_score) < 0.02:
        print("[WARN] predicted score is within 0.02 of a uniform guess -- weak/no signal for this "
              "unseen model on this query. Not conclusive from one query; try several --query_id values.")
    if abs(excl_pred_rank - excl_gt_rank) <= 1:
        print("[OK] predicted rank is within 1 position of ground truth -- consistent with working "
              "inductive generalization (on this one query).")
    else:
        print(f"[WARN] predicted rank is {abs(excl_pred_rank - excl_gt_rank)} positions off ground "
              f"truth -- inspect across more queries before concluding either way.")


if __name__ == "__main__":
    main()