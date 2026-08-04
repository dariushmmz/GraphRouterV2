"""
Core orchestration class for training / inference / k-fold CV of the
multi-task graph router.

Extracted from the original monolithic multi_task_graph_router.py.
Responsibilities that don't need direct access to this class's internal
state have been moved out to sibling modules:

  - router_utility.py   : utility scoring + ranking metrics (pure functions)
  - router_plotting.py  : plotly chart generation (pure functions)
  - io_utils.py          : JSON / pickle load-save helpers (pure functions)
  - data_integrity.py    : DataIntegrityMixin (row/query-block enforcement)
  - kfold_cv.py           : KFoldCVMixin (leakage-free K-fold CV loop)

No behavior changes were made during this split -- every method body is
unchanged from the original file, only relocated and re-imported.
"""

import os
import random

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import torch

from model.graph_data import form_data
from model.graph_layers import EncoderDecoderNet, resolve_checkpoint_dir
# NOTE: run_kfold_cv is no longer imported/used here -- it split already-
# featurized rows internally, which caused the cross-fold encoder-fitting
# leakage described at the kfold_cv branch below. Replaced by
# KFoldCVMixin._run_kfold_cv_no_leakage(), which fits fresh encoders per
# fold before featurizing. run_kfold_cv() itself still exists in
# model/gnn_kfold_legacy.py but is unused here; leave/remove it there as
# you see fit.
from data_processing.utils import ask_and_save_feedback, ensure_2d, parse_embedding_field
from data_processing import feature_builder as fb

from model.data_integrity import DataIntegrityMixin
from model.kfold_cv import KFoldCVMixin
from model.router_utility import (
    calculate_query_utility,
    compute_ranking_metrics,
    resolve_utility_weights,
    auto_drop_zero_variance_utility_weights,
)
from model.router_plotting import make_plot, plot_ground_truth_ranking
from model.io_utils import loadjson, loadpkl
from model.explainability import explain_selection

device = "cuda" if torch.cuda.is_available() else "cpu"
print("---------------> ALL IMPORTED")


class graph_router_prediction(DataIntegrityMixin, KFoldCVMixin):
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
        # Reasoning node feature width = number of domain macro-categories
        # (== number of shared Reasoning prototype nodes). Fixed by the
        # taxonomy in feature_builder.DOMAIN_MACRO_CATEGORIES, not fit from
        # data, so this is stable across train/val/test/inference.
        self.reasoning_dim = self.encoders.reasoning_node_dim
        print(f"[feature_builder] fit on {len(train_df)} train rows -- "
              f"node_metadata_dim={self.encoders.node_metadata_dim}, "
              f"edge_feature_dim={self.encoders.edge_feature_dim} (+1 utility col = {self.edge_dim} total), "
              f"reasoning_dim={self.reasoning_dim} ({self.encoders.reasoning_categories})")

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
            if not config.get('ablation_disable_node_metadata', False) and node_metadata.shape[1] > 0:
                query_embedding = np.concatenate([query_embedding, node_metadata], axis=1)

            query_dim = query_embedding.shape[1]

            # Reasoning prototype nodes (shared, one per domain
            # macro-category) + which one this single query belongs to.
            self.reasoning_features = fb.build_reasoning_node_features(self.encoders)
            self.reasoning_node_id = fb.assign_reasoning_node_index(rows.iloc[[0]], self.encoders)

            self.model = EncoderDecoderNet(query_feature_dim=query_dim, llm_feature_dim=self.llm_dim,
                                           hidden_features=self.config['embedding_dim'], in_edges=self.edge_dim,
                                           reasoning_feature_dim=self.reasoning_dim,
                                           edge_aware=self.config.get('ablation_edge_aware', True)).to(
                device)
            self.form_data = form_data(device)

            # Cache the exact tensors this inference call used. Purely
            # additive (nothing downstream reads these) -- exists so
            # external experiment scripts (e.g. inductive-generalization
            # splice tests) can reuse the IDENTICAL query/task embedding
            # this instance's trained model saw, instead of recomputing
            # them and risking silent drift from this code path.
            #
            # BUGFIX: 'row_id' is a per-(query,model) row identifier -- each
            # model's row for the same query has a DIFFERENT row_id. It is
            # NOT a safe lookup key for "give me this query's row for a
            # different model". The actual query-level identifier is
            # 'query_id' (mirrors the group_col logic in
            # _enforce_rectangular_query_blocks, which falls back to the raw
            # 'query' text column if 'query_id' isn't present). Cache BOTH
            # the group column name and this query's value under it, so
            # callers look up the correct grouping key regardless of which
            # column is actually in use.
            group_col = "query_id" if "query_id" in self.data_df.columns else "query"
            self.last_query_group_col = group_col
            self.last_query_group_value = rows.iloc[0][group_col]
            self.last_query_embedding = query_embedding
            self.last_task_embedding = task_embedding
            self.last_rows_df = rows
            self.last_query_row_id = rows.iloc[0]['row_id']

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
            # K-Fold CV mode.
            #
            # LEAKAGE FIX (was: featurize-once-then-split):
            # The old code called self.prepare_data_for_GNN() ONCE, which
            # featurized the ENTIRE dataset using self.encoders -- an encoder
            # instance fit in __init__ on a single global train split, before
            # this branch even runs. run_kfold_cv() then ran its own KFold
            # over those already-featurized rows. Every fold's "held-out"
            # queries had therefore already influenced (via that earlier,
            # unrelated global split, which generally overlaps with a given
            # fold's holdout) the encoder categories / normalization stats
            # used to featurize them -- exactly backwards from proper CV.
            #
            # Fix: split raw (unfeaturized) query rows into folds FIRST, then
            # -- independently, per fold -- fit a fresh FeatureEncoders
            # instance on ONLY that fold's inner-train rows and use it
            # (transform-only) for that fold's inner-val and holdout rows.
            # No encoder instance is ever shared across folds or fit on
            # anything outside its own fold's inner-train split. This is
            # implemented in KFoldCVMixin._run_kfold_cv_no_leakage(); it
            # replaces the old prepare_data_for_GNN()+run_kfold_cv() call
            # entirely, since run_kfold_cv()'s internal splitting operated
            # on already-featurized tensors and can't be made leakage-safe
            # without moving encoder fitting inside its loop.
            self.kfold_results = self._run_kfold_cv_no_leakage()
        else:
            from model.gnn_trainer import GNN_prediction
            self.prepare_data_for_GNN()
            self.split_data()
            self.form_data = form_data(device)
            self.query_dim = self.query_embedding_list.shape[1]
            self.GNN_predict = GNN_prediction(query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim,
                                              hidden_features_size=self.config['embedding_dim'],
                                              in_edges_size=self.edge_dim,
                                              reasoning_feature_dim=self.reasoning_dim,
                                              wandb=self.wandb, config=self.config,
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
        # Question Metadata ablation: skip the concat entirely when disabled.
        # query_feature_dim downstream (self.query_dim, line ~329) is
        # inferred from self.query_embedding_list.shape[1] at runtime, so no
        # other dimension needs updating -- this is a clean toggle. Must
        # match whatever this checkpoint's training run used (see config.yaml
        # comment on ablation_disable_node_metadata) or load_state_dict will
        # fail on a shape mismatch at inference time.
        if not self.config.get('ablation_disable_node_metadata', False) and node_metadata.shape[1] > 0:
            self.query_embedding_list = np.concatenate(
                [self.query_embedding_list, node_metadata], axis=1
            )

        # --- Reasoning node type: shared prototype nodes (one per domain
        # macro-category, NOT one per query) plus each query's assignment
        # to one of those prototypes. Domain macro-category is deliberately
        # excluded from node_metadata above -- it lives on these Reasoning
        # nodes instead, connected to Question nodes via edges built in
        # form_data.formulation(). ---
        self.reasoning_features = fb.build_reasoning_node_features(self.encoders)
        self.reasoning_node_id = fb.assign_reasoning_node_index(self.data_df, self.encoders)[unique_index_list]

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
                                                             test_mask=self.mask_test,
                                                             reasoning_feature=self.reasoning_features,
                                                             reasoning_node_id=self.reasoning_node_id)
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
                                                                test_mask=self.mask_test,
                                                                reasoning_feature=self.reasoning_features,
                                                                reasoning_node_id=self.reasoning_node_id)

        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.utility_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test,
                                                        reasoning_feature=self.reasoning_features,
                                                        reasoning_node_id=self.reasoning_node_id)
        self.GNN_predict.train_validate(data=self.data_for_GNN_train, data_validate=self.data_for_GNN_validate,
                                        data_for_test=self.data_for_test)

    def test_GNN(self):
        # Consistent with infer_single_query/train_validate -- resolve the
        # ablation-arm-specific path rather than the raw config value, even
        # though GNN_prediction.test()'s model_path arg is currently unused
        # internally (it scores self.model already in memory). Keeping it
        # accurate here avoids a misleading value if that changes later.
        predicted_result = self.GNN_predict.test(
            data=self.data_for_test, model_path=resolve_checkpoint_dir(self.config)
        )
        return predicted_result

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
        # Ablation-aware resolution: resolves to the SAME path training used
        # for these exact ablation flags (see resolve_checkpoint_dir() in
        # graph_layers.py) -- unchanged from plain config['model_path'] when
        # both ablation flags are at their defaults.
        checkpoint_dir = resolve_checkpoint_dir(self.config)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        ablation_desc = (
            f"ablation_disable_node_metadata="
            f"{self.config.get('ablation_disable_node_metadata', False)}, "
            f"ablation_edge_aware={self.config.get('ablation_edge_aware', True)}"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint found at '{checkpoint_path}' for the currently "
                f"active ablation settings ({ablation_desc}). Each ablation arm "
                f"needs its own trained checkpoint -- run training with these "
                f"exact config values first (it will save here automatically), "
                f"or reset the ablation flags to their defaults to use the "
                f"checkpoint at "
                f"'{os.path.join(self.config['model_path'], 'best_model.pth')}'."
            )
        try:
            loaded_state = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to read checkpoint file at '{checkpoint_path}': {e}") from e

        # Pre-flight ablation compatibility check: compares the checkpoint's
        # ACTUAL saved parameter shapes/keys against what the currently
        # active config implies, and raises a specific, actionable
        # diagnostic (naming the exact dims and which flag is inconsistent)
        # instead of surfacing PyTorch's generic
        # "size mismatch...copying a param with shape [...]" from
        # load_state_dict, which gives no indication of *why*.
        current_query_dim = np.array(query_embedding).reshape(1, -1).shape[1]
        ckpt_query_w = loaded_state.get('model_align.query_transform.0.weight')
        if ckpt_query_w is not None and ckpt_query_w.shape[1] != current_query_dim:
            ckpt_had_metadata = ckpt_query_w.shape[1] > current_query_dim
            raise RuntimeError(
                f"Checkpoint/config mismatch at '{checkpoint_path}':\n"
                f"  checkpoint's query_transform expects input dim "
                f"{ckpt_query_w.shape[1]}, but the current config builds a "
                f"query embedding of dim {current_query_dim}.\n"
                f"  This checkpoint was trained with node_metadata "
                f"{'INCLUDED' if ckpt_had_metadata else 'EXCLUDED'} "
                f"(ablation_disable_node_metadata="
                f"{not ckpt_had_metadata}), but the active config has "
                f"ablation_disable_node_metadata="
                f"{self.config.get('ablation_disable_node_metadata', False)}.\n"
                f"Fix: either set ablation_disable_node_metadata="
                f"{not ckpt_had_metadata} to match this checkpoint, or train "
                f"a fresh checkpoint for the current ablation setting -- it "
                f"will save to this same path via resolve_checkpoint_dir()."
            )

        ckpt_edge_aware = 'edge_mlp.weight' in loaded_state
        config_edge_aware = self.config.get('ablation_edge_aware', True)
        if ckpt_edge_aware != config_edge_aware:
            raise RuntimeError(
                f"Checkpoint/config mismatch at '{checkpoint_path}':\n"
                f"  checkpoint was trained with ablation_edge_aware="
                f"{ckpt_edge_aware}, but the active config has "
                f"ablation_edge_aware={config_edge_aware}.\n"
                f"Fix: either set ablation_edge_aware={ckpt_edge_aware} to "
                f"match this checkpoint, or train a fresh checkpoint for the "
                f"current ablation setting -- it will save to this same path "
                f"via resolve_checkpoint_dir()."
            )

        try:
            self.model.load_state_dict(loaded_state)
        except RuntimeError as e:
            # Should be rare now that the two checks above catch the common
            # ablation-flag-mismatch cases explicitly -- if this still fires,
            # it's something other than the flags covered above (e.g. a
            # genuinely stale/foreign checkpoint file at this path).
            raise RuntimeError(
                f"Checkpoint at '{checkpoint_path}' does not match the model "
                f"architecture implied by the current ablation settings "
                f"({ablation_desc}), for a reason other than the "
                f"node_metadata/edge_aware dims checked above -- inspect "
                f"the checkpoint's other layer shapes directly.\n"
                f"Original error: {e}"
            ) from e
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
            test_mask=test_mask,
            reasoning_feature=self.reasoning_features,
            reasoning_node_id=self.reasoning_node_id
        )

        # During inference, allow full visibility
        edge_can_see = torch.ones(num_llms).bool().to(device)

        # ---- Forward ----
        with torch.no_grad():
            edge_scores = self.model(
                task_id=data.task_id,
                query_features=data.query_features,
                llm_features=data.llm_features,
                reasoning_features=data.reasoning_features,
                edge_index=data.edge_index,
                reasoning_edge_index=data.reasoning_edge_index,
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

        # --- Explainability: lightweight, rule-based reasons for the pick ---
        # Uses the exact utility_weights already driving training/ranking
        # (router_utility.py) plus, when available, the same rows_df raw
        # Correct/Cost/Latency/Completion_Status/Output_Tokens columns
        # already used for ranking_metrics below. Falls back to a
        # predicted-score-margin-only explanation when rows_df isn't
        # available (true blind inference on an unseen query has no raw
        # outcome data to compare against).
        explanation = explain_selection(
            selected_model=result['best_llm'],
            scores=scores,
            weights=self.utility_weights,
            rows_df=rows_df,
            model_col='model',
            id_to_display=getattr(self, 'router_id_to_display', None),
        )
        result['explanation'] = explanation['text']
        result['explanation_signals'] = explanation['signals']
        print("\n" + explanation['text'])

        # --- Metrics between predicted ranking and ground-truth ranking ---
        # Uses calculate_query_utility() (raw Correct/Cost/Latency/Output_Tokens/
        # Completion_Status columns, NOT the softmax-smoothed edge_attr used
        # above) as the ground truth, since that's the actual utility scale
        # the router is meant to be judged against.
        if query_row_id is not None and rows_df is not None:
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

        return result