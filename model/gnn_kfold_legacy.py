"""
Query-level K-Fold cross validation driver for GNN_prediction.

Extracted from graph_nn.py -- no logic changes, only relocation.

NOTE: as flagged in multi_task_graph_router.py's own comments, this
run_kfold_cv() function operates on ALREADY-FEATURIZED rows (encoders
fit once, globally, before this function's fold split), which is a
cross-fold leakage source. It has been superseded by
model.kfold_cv.KFoldCVMixin._run_kfold_cv_no_leakage(), which fits fresh
encoders per fold before featurizing. This function is kept here,
unused by the current pipeline, in case you still want it for
reference/comparison -- remove it if not.
"""

import os
import shutil

import numpy as np
import torch
from sklearn.model_selection import KFold

from model.graph_data import form_data
from model.gnn_trainer import GNN_prediction

try:
    from router_evaluation import build_oof_predictions_df, evaluate_router_performance
except ImportError:
    build_oof_predictions_df = evaluate_router_performance = None


def run_kfold_cv(query_feature_dim, llm_feature_dim, hidden_features_size, in_edges_size,
                  config, device, task_embedding_list, query_embedding_list,
                  llm_description_embedding, edge_org_id, edge_des_id, utility_list,
                  label, combined_edge, num_llms, num_query, k=5, inner_val_ratio=0.1,
                  wandb=None, make_plots=False, data_df=None, gold_col="utility",
                  cost_col="Cost", correctness_col="correctness", latency_col=None):
    """
    Query-level K-Fold cross validation.

    Rationale: with ~71 total queries, a single 90/10-style train/val/test
    split leaves only ~7 validation queries -- any single split's metrics are
    dominated by which particular queries happened to land in that split.
    K-Fold rotates every query through the held-out role exactly once, so the
    reported metrics average out that single-split variance and give a much
    more trustworthy signal for whether an architecture/config change is
    actually helping.

    Splitting is done at the QUERY level (not row level): each query owns
    exactly `num_llms` consecutive rows, and all rows for a query must stay
    together in the same fold, otherwise edges from the same query would leak
    across train/held-out boundaries.

    For each of the k folds:
      - the fold's queries are the held-out set (used as this fold's
        validation set for reporting the CV score)
      - a small slice of the remaining (non-held-out) queries is carved out
        as an inner validation set, used only for checkpoint selection
        (best val_loss) and early stopping, exactly as in train_validate
      - the rest of the remaining queries are the training set
      - a FRESH model/optimizer/scheduler is trained from scratch for this
        fold (folds must not share weights, or later folds leak information
        from queries held out in earlier folds)

    Returns a dict with per-fold metrics and their mean/std, and saves each
    fold's best checkpoint under `<model_path>/fold_<i>/best_model.pth`. The
    single best checkpoint across all folds (by the same lowest-val_loss
    criterion used for each fold's own checkpointing) is additionally copied
    to `<model_path>/best_model.pth`, at the root -- not inside any
    `fold_<i>/` subdirectory -- so there's one obvious "best overall" weights
    file to load for actual inference, alongside the per-fold ones kept for
    inspection/reproducibility.

    If `data_df` is provided (the original rectangular router_data
    DataFrame, one row per query-model edge, same order as edge_org_id/
    edge_des_id/utility_list), this also pools every fold's held-out
    predictions into a single out-of-fold table and reports the full
    router_evaluation suite (Relative Utility Gap Closed vs Oracle/SBM/
    Random/Cost-Optimal baselines, Top-1 / Top-K match, NDCG@3, cost &
    accuracy retention vs SBM) under the returned 'router_eval' key. This
    is the recommended way to read out overall router quality on a small
    (~71-query) dataset: it's a single pooled score over every query
    (each scored by the model that had it held out) rather than an
    average of five noisy per-fold accuracies, and Gap Closed in
    particular gives partial credit for near-ties and penalizes bad
    misses proportionally to how much utility they actually cost --
    unlike Top-1 accuracy on its own.
    """
    formatter = form_data(device)
    base_model_path = config['model_path']

    rng = np.random.RandomState(config.get('seed', 0))
    query_indices = np.arange(num_query)
    rng.shuffle(query_indices)

    kf = KFold(n_splits=k, shuffle=False)  # already shuffled above, deterministically

    def rows_for_queries(q_idx):
        rows = []
        for q in q_idx:
            start = q * num_llms
            rows.extend(range(start, start + num_llms))
        return rows

    fold_results = []
    oof_predicted_utility = np.full(len(edge_org_id), np.nan, dtype=float)

    # Best checkpoint ACROSS all folds, selected by the same criterion each
    # fold already uses internally (lowest val_loss at its own best epoch),
    # copied into the root model_path (not a fold_<i>/ subdirectory) so
    # there's a single, easy-to-find "best overall" weights file alongside
    # the per-fold ones.
    best_overall_val_loss = float('inf')
    best_overall_fold = None
    best_overall_path = os.path.join(base_model_path, "best_model.pth")

    for fold_i, (remain_pos, holdout_pos) in enumerate(kf.split(query_indices)):
        remain_queries = query_indices[remain_pos]
        holdout_queries = query_indices[holdout_pos]

        # Carve a small inner-validation slice out of the remaining queries,
        # used only for checkpoint/early-stopping -- never for the reported
        # CV metric, so the held-out fold stays a clean, unused-for-selection
        # estimate of generalization.
        rng.shuffle(remain_queries)
        n_inner_val = max(1, int(len(remain_queries) * inner_val_ratio))
        inner_val_queries = remain_queries[:n_inner_val]
        train_queries = remain_queries[n_inner_val:]

        train_row_idx = rows_for_queries(train_queries)
        inner_val_row_idx = rows_for_queries(inner_val_queries)
        holdout_row_idx = rows_for_queries(holdout_queries)

        mask_train = torch.zeros(len(edge_org_id))
        mask_train[train_row_idx] = 1
        mask_inner_val = torch.zeros(len(edge_org_id))
        mask_inner_val[inner_val_row_idx] = 1
        mask_holdout = torch.zeros(len(edge_org_id))
        mask_holdout[holdout_row_idx] = 1

        data_train = formatter.formulation(
            task_id=task_embedding_list, query_feature=query_embedding_list,
            llm_feature=llm_description_embedding, org_node=edge_org_id, des_node=edge_des_id,
            edge_feature=utility_list, edge_mask=mask_train, label=label, combined_edge=combined_edge,
            train_mask=mask_train, valide_mask=mask_inner_val, test_mask=mask_holdout)

        data_inner_val = formatter.formulation(
            task_id=task_embedding_list, query_feature=query_embedding_list,
            llm_feature=llm_description_embedding, org_node=edge_org_id, des_node=edge_des_id,
            edge_feature=utility_list, edge_mask=mask_inner_val, label=label, combined_edge=combined_edge,
            train_mask=mask_train, valide_mask=mask_inner_val, test_mask=mask_holdout)

        data_holdout = formatter.formulation(
            task_id=task_embedding_list, query_feature=query_embedding_list,
            llm_feature=llm_description_embedding, org_node=edge_org_id, des_node=edge_des_id,
            edge_feature=utility_list, edge_mask=mask_holdout, label=label, combined_edge=combined_edge,
            train_mask=mask_train, valide_mask=mask_inner_val, test_mask=mask_holdout)

        fold_config = dict(config)
        fold_config['model_path'] = os.path.join(base_model_path, f"fold_{fold_i}")

        gnn = GNN_prediction(
            query_feature_dim=query_feature_dim, llm_feature_dim=llm_feature_dim,
            hidden_features_size=hidden_features_size, in_edges_size=in_edges_size,
            config=fold_config, device=device, wandb=wandb)

        print(f"\n===== Fold {fold_i + 1}/{k} "
              f"(train={len(train_queries)}, inner_val={len(inner_val_queries)}, "
              f"held_out={len(holdout_queries)} queries) =====")

        gnn.train_validate(data=data_train, data_validate=data_inner_val, data_for_test=data_holdout,
                            make_plots=make_plots)

        # Evaluate this fold's best checkpoint on its held-out queries -- this
        # is the number that actually goes into the CV average.
        holdout_result, holdout_loss = gnn.test(data_holdout, gnn.save_path)

        # Load this fold's best checkpoint (test() above already re-runs the
        # forward pass but only returns the scalar summary) and stash the
        # RAW per-row predictions for this fold's held-out rows into the
        # pooled OOF buffer -- every query in the dataset ends up scored by
        # exactly the fold that had it held out, with no leakage.
        gnn.model.load_state_dict(torch.load(gnn.save_path, map_location=device))
        holdout_row_idx_sorted = sorted(holdout_row_idx)
        oof_predicted_utility[holdout_row_idx_sorted] = gnn.predict_raw(data_holdout)

        best_epoch = int(np.argmin(gnn.history['val_loss']))
        fold_best_val_loss = gnn.history['val_loss'][best_epoch]
        fold_results.append({
            'fold': fold_i,
            'best_val_loss': fold_best_val_loss,
            'val_f1_at_best_loss': gnn.history['val_f1'][best_epoch],
            'val_accuracy_at_best_loss': gnn.history['val_accuracy'][best_epoch],
            'holdout_loss': gnn._to_numpy(holdout_loss),
            'holdout_predict_mean': gnn._to_numpy(holdout_result),
        })

        # Same checkpoint-selection criterion train_validate() uses per-fold
        # (lowest val_loss), applied across folds: if this fold's best
        # checkpoint beats every fold seen so far, copy it up to the root
        # model_path as best_model.pth. gnn.save_path is guaranteed to exist
        # on disk at this point -- train_validate() already wrote it.
        if fold_best_val_loss < best_overall_val_loss:
            best_overall_val_loss = fold_best_val_loss
            best_overall_fold = fold_i
            os.makedirs(base_model_path, exist_ok=True)
            shutil.copy2(gnn.save_path, best_overall_path)
            print(f"[kfold] fold {fold_i} is the new best overall (val_loss="
                  f"{fold_best_val_loss:.4f}) -- copied to {best_overall_path}")

    metrics_keys = ['best_val_loss', 'val_f1_at_best_loss', 'val_accuracy_at_best_loss',
                     'holdout_loss', 'holdout_predict_mean']
    summary = {}
    for key in metrics_keys:
        values = np.array([r[key] for r in fold_results], dtype=float)
        summary[key] = {'mean': float(values.mean()), 'std': float(values.std())}

    print("\n========== K-Fold Cross Validation Summary ==========")
    for key in metrics_keys:
        print(f"{key}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
    print(f"Best overall fold: {best_overall_fold} (val_loss={best_overall_val_loss:.4f}) "
          f"-- weights saved to {best_overall_path}")

    router_eval = None
    if data_df is not None:
        if build_oof_predictions_df is None:
            print("[router_eval][WARN] data_df was passed but router_evaluation.py "
                  "isn't importable -- skipping pooled Gap Closed / NDCG report. "
                  "Make sure router_evaluation.py is on the path.")
        elif np.isnan(oof_predicted_utility).any():
            n_missing = int(np.isnan(oof_predicted_utility).sum())
            print(f"[router_eval][WARN] {n_missing} row(s) never got an out-of-fold prediction "
                  f"(fold/query-count mismatch?) -- skipping pooled router_eval report.")
        else:
            oof_df, cols = build_oof_predictions_df(
                data_df=data_df, num_llms=num_llms, oof_predicted_utility=oof_predicted_utility,
                gold_col=gold_col, cost_col=cost_col, correctness_col=correctness_col,
                latency_col=latency_col)
            print("\n========== Pooled Out-of-Fold Router Evaluation ==========")
            router_eval = evaluate_router_performance(
                oof_df, pred_col="predicted_utility", gold_col=cols['gold_col'],
                query_id_col=cols['query_id_col'], cost_col=cols['cost_col'] or "Cost",
                correctness_col=cols['correctness_col'] or "correctness",
                latency_col=cols['latency_col'])

    return {
        'folds': fold_results,
        'summary': summary,
        'router_eval': router_eval,
        'best_overall_fold': best_overall_fold,
        'best_overall_val_loss': best_overall_val_loss,
        'best_overall_model_path': best_overall_path,
    }
