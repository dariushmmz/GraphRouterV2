"""
Row/query integrity enforcement for the graph router's positional-indexing
assumptions.

Extracted from multi_task_graph_router.py -- no logic changes, only
relocation. This stays a mixin (rather than becoming pure functions)
because the original methods read/write a dozen `self.*` attributes
(self.num_llms, self.llm_names, self.config, self.train_row_idx, ...)
that are set up across __init__ and consumed elsewhere in the class.
Converting it to pure functions would mean threading all of those through
explicit parameters and return tuples, which risks changing behavior
without being able to test against the real graph_nn.py / feature_builder.py
pipeline. GraphRouterPrediction inherits this mixin instead.
"""

import pandas as pd


class DataIntegrityMixin:
    """
    Provides:
      - _compute_split_row_indices(): train/val/test row index computation.
      - _enforce_rectangular_query_blocks(): drops/reorders incomplete or
        duplicate query blocks so every query occupies exactly num_llms
        consecutive rows in canonical model order.
    """

    def _compute_split_row_indices(self):
        """
        Row indices (into self.data_df, 0-indexed, task/query/llm-major order)
        for train/validate/test. Extracted out of split_data() so it can run
        BEFORE encoder fitting -- encoders must only see train rows.
        Idempotent: safe to call more than once (e.g. once early for encoder
        fitting, once again implicitly via split_data() for mask construction).
        """
        self.query_per_task = int(self.num_query / self.num_task)
        split_ratio = self.config['split_ratio']

        train_size = int(self.query_per_task * split_ratio[0])
        val_size = int(self.query_per_task * split_ratio[1])
        test_size = int(self.query_per_task * split_ratio[2])

        train_idx, validate_idx, test_idx = [], [], []
        for task_id in range(self.num_task):
            start_idx = task_id * self.query_per_task * self.num_llms
            train_idx.extend(range(start_idx, start_idx + train_size * self.num_llms))
            validate_idx.extend(range(start_idx + train_size * self.num_llms,
                                      start_idx + train_size * self.num_llms + val_size * self.num_llms))
            test_idx.extend(range(start_idx + train_size * self.num_llms + val_size * self.num_llms,
                                  start_idx + train_size * self.num_llms + val_size * self.num_llms + test_size * self.num_llms))

        self.train_row_idx = train_idx
        self.validate_row_idx = validate_idx
        self.test_row_idx = test_idx
        return train_idx, validate_idx, test_idx

    def _enforce_rectangular_query_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantee the invariant every downstream positional index relies on:
        each query occupies exactly self.num_llms consecutive rows, one per
        distinct model value found in the router data. Groups by `query_id`
        (present in router_data.csv per Revised_Feature_Plan.md); falls back
        to grouping by raw `query` text with a warning if query_id is missing.

        The "complete set" of models is derived from the router dataset's own
        `model` column (self-consistency check), NOT compared against
        LLM_Descriptions.json's llm_names -- the two use unrelated naming
        schemes and matching them is a separate concern from data integrity.
        The one thing we do verify is that the router data's own model
        vocabulary has exactly self.num_llms distinct values, since that's
        the cardinality every downstream tensor shape assumes.

        Any query_id whose row set doesn't cover that vocabulary exactly once
        (missing run, duplicate run) is DROPPED WHOLESALE and reported --
        keeping it would silently misalign edge_org_id/edge_des_id/
        unique_index_list for every query after it, which is strictly worse
        than losing that one query's data.
        """
        group_col = "query_id" if "query_id" in df.columns else "query"
        if group_col == "query":
            print("[data_integrity][WARN] no query_id column found -- grouping by raw "
                  "query text instead. If two different queries have identical text this "
                  "will incorrectly merge them.")

        wanted_models = sorted(df["model"].unique().tolist())
        if len(wanted_models) != self.num_llms:
            raise ValueError(
                f"[data_integrity] router_data's `model` column has {len(wanted_models)} distinct "
                f"values {wanted_models}, but LLM_Descriptions.json declares {self.num_llms} LLMs "
                f"({self.llm_names}). These counts must match -- check for typos/extra models in "
                f"the data or a stale LLM_Descriptions.json."
            )
        wanted_models_set = set(wanted_models)
        llm_order = {name: i for i, name in enumerate(wanted_models)}  # stable order, data-derived

        # BUGFIX (name mismatch): `wanted_models` (raw router ids, e.g.
        # 'openai/gpt-5') is what every row gets positionally sorted into,
        # and self.llm_names (display names, e.g. 'GPT-5') is what the rest
        # of the pipeline (scores dict, best_llm, etc.) is keyed by. Nothing
        # upstream ever recorded which raw id corresponds to which display
        # name -- the two lists just happen to share the same positional
        # order once both are the index-i-th model. Persist that
        # correspondence here, once, so downstream code (infer_single_query /
        # compute_ranking_metrics) can translate router ids <-> display
        # names instead of comparing them directly.
        self.router_model_ids = wanted_models
        self.router_id_to_display = dict(zip(self.router_model_ids, self.llm_names))

        kept_blocks = []
        n_dropped_queries = 0
        dropped_ids = []
        for key, group in df.groupby(group_col, sort=False):
            models_here = group["model"].tolist()
            if len(models_here) != self.num_llms or set(models_here) != wanted_models_set:
                n_dropped_queries += 1
                dropped_ids.append(key)
                continue
            kept_blocks.append(group.assign(_llm_sort=group["model"].map(llm_order)).sort_values("_llm_sort"))

        if n_dropped_queries:
            preview = dropped_ids[:10]
            print(f"[data_integrity][WARN] dropped {n_dropped_queries} incomplete/duplicate "
                  f"query group(s) out of {df[group_col].nunique()} -- these queries did not "
                  f"have exactly one row per model in {wanted_models}. Example query_id(s): {preview}"
                  f"{' ...' if n_dropped_queries > 10 else ''}")

        if not kept_blocks:
            raise ValueError(
                "[data_integrity] 0 complete query blocks after filtering -- check the `model` "
                "column values in router_data.csv for typos/inconsistent naming: " + str(wanted_models)
            )

        result = pd.concat(kept_blocks, axis=0, ignore_index=True).drop(columns=["_llm_sort"])
        n_before, n_after = len(df), len(result)
        print(f"[data_integrity] rectangular-block check: {n_before} -> {n_after} rows "
              f"({n_before - n_after} dropped), {n_after // self.num_llms} complete queries retained.")
        return result
