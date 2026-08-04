"""
Leakage-free K-Fold cross-validation for the graph router.

Extracted from multi_task_graph_router.py -- no logic changes, only
relocation. Stays a mixin for the same reason as data_integrity.py: these
methods read self.data_df / self.num_llms / self.config / self.llm_dim /
self.llm_description_embedding / self.wandb / self.utility_weights and
write into locally-scoped fold state. GraphRouterPrediction inherits this
mixin instead of standing it up as free functions, to avoid threading ~10
parameters through every call without being able to test against the real
pipeline.
"""

import os
import shutil

import numpy as np
import torch

from data_processing.utils import parse_embedding_field
from data_processing import feature_builder as fb

device = "cuda" if torch.cuda.is_available() else "cpu"


class KFoldCVMixin:
    """
    Provides:
      - _rows_for_query_indices(): maps query-block indices to raw row indices.
      - _featurize_fold_df(): featurizes one fold's rows with fold-local encoders.
      - _run_kfold_cv_no_leakage(): the full leakage-free K-fold CV loop.
    """

    def _rows_for_query_indices(self, query_indices):
        """
        Map a list/array of query indices (0-indexed, into the num_llms-row
        blocks of self.data_df) to the concrete raw row indices for those
        queries. Relies on the rectangular-block invariant already enforced
        by _enforce_rectangular_query_blocks() (each query = exactly
        self.num_llms consecutive rows).
        """
        rows = []
        for q in query_indices:
            start = int(q) * self.num_llms
            rows.extend(range(start, start + self.num_llms))
        return rows

    def _featurize_fold_df(self, fold_df, encoders):
        """
        Build query/task embeddings, node metadata, edge features, and the
        per-query-softmaxed utility/label for a single dataframe (fold_df),
        using the given (already-fit) `encoders` in transform-only fashion.
        Mirrors prepare_data_for_GNN()/split_data()'s featurization logic,
        factored out so it can be called once per fold with fold-specific
        encoders instead of once globally with self.encoders.
        """
        unique_index_list = list(range(0, len(fold_df), self.num_llms))

        query_embedding_list = np.array(
            [parse_embedding_field(e) for e in fold_df['query_embedding'].tolist()]
        )[unique_index_list]
        task_embedding_list = np.array(
            [parse_embedding_field(e) for e in fold_df['task_description_embedding'].tolist()]
        )[unique_index_list]

        node_metadata = fb.build_node_metadata(fold_df, encoders)[unique_index_list]
        if node_metadata.shape[1] > 0:
            query_embedding_list = np.concatenate([query_embedding_list, node_metadata], axis=1)

        # Reasoning node type: shared prototype nodes (one per domain
        # macro-category) built from this fold's own fold-local encoders --
        # same leakage-free treatment as everything else here, since
        # reasoning_categories is derived from `encoders`, not self.encoders.
        reasoning_features = fb.build_reasoning_node_features(encoders)
        reasoning_node_id = fb.assign_reasoning_node_index(fold_df, encoders)[unique_index_list]

        edge_features = fb.build_edge_features(fold_df, encoders)
        utility_list = fb.compute_utility(fold_df, encoders, **self.utility_weights)

        # Same per-query softmax + one-hot label derivation as split_data().
        SOFTMAX_TEMPERATURE = 1.0
        utility_reshaped = utility_list.reshape(-1, self.num_llms) / SOFTMAX_TEMPERATURE
        exp_scores = np.exp(utility_reshaped - np.max(utility_reshaped, axis=1, keepdims=True))
        softmax_per_query = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        utility_softmaxed = softmax_per_query.reshape(-1)

        utility_re = utility_softmaxed.reshape(-1, self.num_llms)
        label = np.eye(self.num_llms)[np.argmax(utility_re, axis=1)].reshape(-1, 1)

        return (query_embedding_list, task_embedding_list, edge_features, utility_softmaxed, label,
                reasoning_features, reasoning_node_id)

    def _run_kfold_cv_no_leakage(self):
        """
        Leakage-free K-Fold CV.

        Splits at the RAW-ROW level (query granularity) before any
        featurization happens. For each fold:
          1. holdout queries are set aside untouched.
          2. the remaining ("train-side") queries for that fold are split
             again into inner-train / inner-val.
          3. a brand-new FeatureEncoders instance is fit on ONLY that fold's
             inner-train rows (fb.fit_encoders) -- never on inner-val or
             holdout rows, and never reused from another fold or from
             self.encoders.
          4. that fold-local encoder is used, transform-only, to featurize
             the fold's inner-train, inner-val, AND holdout rows.
          5. a fresh GNN_prediction model is trained on the fold's
             inner-train/inner-val split and evaluated on its holdout.

        This guarantees no fold's held-out rows ever influence the encoder
        categories / normalization statistics used to featurize them --
        the leakage present in the old featurize-once-then-KFold-split
        approach (see call site comment in __init__).
        """
        from sklearn.model_selection import KFold
        from model.gnn_trainer import GNN_prediction
        from model.graph_data import form_data

        k = self.config.get('kfold_k', 5)
        inner_val_ratio = self.config.get('kfold_inner_val_ratio', 0.1)
        difficulty_order = self.config.get('difficulty_ordinal_order')
        reasoning_depth_order = self.config.get('reasoning_depth_ordinal_order')

        query_ids = np.arange(self.num_query)
        kf = KFold(n_splits=k, shuffle=True, random_state=self.config['seed'])

        fold_results = []
        # Best checkpoint ACROSS all folds, selected the same way each fold
        # already selects its own checkpoint internally (lowest val_loss at
        # its own best epoch) -- copied into the root config['model_path']
        # (not a fold_<i>/ subdirectory) so there's a single obvious
        # "best overall" weights file to load for real inference, alongside
        # the per-fold ones kept for inspection/reproducibility.
        best_overall_val_loss = float('inf')
        best_overall_fold = None
        best_overall_path = os.path.join(self.config['model_path'], "best_model.pth")

        for fold_i, (trainval_q, holdout_q) in enumerate(kf.split(query_ids)):
            print(f"\n========== Fold {fold_i + 1}/{k} (no-leakage) ==========")

            # Inner train/val split of THIS FOLD's train-side queries only.
            # holdout_q is never touched here or in encoder fitting below.
            rng = np.random.RandomState(self.config['seed'] + fold_i)
            shuffled = trainval_q.copy()
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * inner_val_ratio)))
            inner_val_q = shuffled[:n_val]
            inner_train_q = shuffled[n_val:]

            train_rows = self._rows_for_query_indices(sorted(inner_train_q.tolist()))
            val_rows = self._rows_for_query_indices(sorted(inner_val_q.tolist()))
            holdout_rows = self._rows_for_query_indices(sorted(holdout_q.tolist()))

            # inner-train rows first, inner-val next, holdout last -- order
            # matters, since edge_org_id/edge_des_id below assume num_llms-
            # consecutive blocks per query in that order.
            fold_df = self.data_df.iloc[train_rows + val_rows + holdout_rows].reset_index(drop=True)
            n_train_rows, n_val_rows, n_holdout_rows = len(train_rows), len(val_rows), len(holdout_rows)

            # --- Fit encoders on THIS FOLD's inner-train rows ONLY. ---
            fold_encoders = fb.fit_encoders(
                fold_df.iloc[:n_train_rows],
                difficulty_ordinal_order=difficulty_order,
                reasoning_depth_ordinal_order=reasoning_depth_order,
            )
            fold_edge_dim = 1 + fold_encoders.edge_feature_dim
            fold_reasoning_dim = fold_encoders.reasoning_node_dim

            # --- Featurize the whole fold (train+val+holdout), transform-
            # only, using ONLY fold_encoders. ---
            (query_embedding_list, task_embedding_list, edge_features,
             utility_softmaxed, label, reasoning_features,
             reasoning_node_id) = self._featurize_fold_df(fold_df, fold_encoders)

            fold_num_query = len(fold_df) // self.num_llms
            edge_org_id = [num for num in range(fold_num_query) for _ in range(self.num_llms)]
            edge_des_id = list(range(self.num_llms)) * fold_num_query

            mask_train = torch.zeros(len(edge_org_id))
            mask_train[:n_train_rows] = 1
            mask_validate = torch.zeros(len(edge_org_id))
            mask_validate[n_train_rows:n_train_rows + n_val_rows] = 1
            mask_test = torch.zeros(len(edge_org_id))
            mask_test[n_train_rows + n_val_rows:n_train_rows + n_val_rows + n_holdout_rows] = 1

            form = form_data(device)
            common_kwargs = dict(
                task_id=task_embedding_list, query_feature=query_embedding_list,
                llm_feature=self.llm_description_embedding, org_node=edge_org_id, des_node=edge_des_id,
                edge_feature=utility_softmaxed, label=label, combined_edge=edge_features,
                train_mask=mask_train, valide_mask=mask_validate, test_mask=mask_test,
                reasoning_feature=reasoning_features, reasoning_node_id=reasoning_node_id,
            )
            data_train = form.formulation(edge_mask=mask_train, **common_kwargs)
            data_val = form.formulation(edge_mask=mask_validate, **common_kwargs)
            data_test = form.formulation(edge_mask=mask_test, **common_kwargs)

            query_dim = query_embedding_list.shape[1]
            fold_model_path = os.path.join(self.config['model_path'], f"fold_{fold_i}")
            fold_config = dict(self.config)
            fold_config['model_path'] = fold_model_path

            gnn = GNN_prediction(
                query_feature_dim=query_dim, llm_feature_dim=self.llm_dim,
                hidden_features_size=self.config['embedding_dim'], in_edges_size=fold_edge_dim,
                reasoning_feature_dim=fold_reasoning_dim,
                wandb=self.wandb, config=fold_config, device=device)

            gnn.train_validate(data=data_train, data_validate=data_val, data_for_test=data_test, fold=fold_i)
            fold_test_result = gnn.test(data=data_test, model_path=fold_model_path)

            fold_best_epoch = int(np.argmin(gnn.history['val_loss']))
            fold_best_val_loss = gnn.history['val_loss'][fold_best_epoch]

            fold_results.append({
                'fold': fold_i,
                'n_train_queries': len(inner_train_q),
                'n_val_queries': len(inner_val_q),
                'n_holdout_queries': len(holdout_q),
                'edge_dim': fold_edge_dim,
                'reasoning_dim': fold_reasoning_dim,
                'best_val_loss': fold_best_val_loss,
                'test_result': fold_test_result,
            })

            # Same checkpoint-selection criterion train_validate() uses
            # per-fold (lowest val_loss): if this fold's best checkpoint
            # beats every fold seen so far, copy it up to the root
            # config['model_path'] as best_model.pth. gnn.save_path is
            # guaranteed to exist on disk here -- train_validate() already
            # wrote it.
            if fold_best_val_loss < best_overall_val_loss:
                best_overall_val_loss = fold_best_val_loss
                best_overall_fold = fold_i
                os.makedirs(self.config['model_path'], exist_ok=True)
                shutil.copy2(gnn.save_path, best_overall_path)
                print(f"[kfold] fold {fold_i} is the new best overall (val_loss="
                      f"{fold_best_val_loss:.4f}) -- copied to {best_overall_path}")

        print("\n========== K-Fold Cross Validation Complete ==========")
        print(f"Best overall fold: {best_overall_fold} (val_loss={best_overall_val_loss:.4f}) "
              f"-- weights saved to {best_overall_path}")

        return fold_results