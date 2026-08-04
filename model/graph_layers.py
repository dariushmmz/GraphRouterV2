"""
Core neural network modules for the graph router: feature alignment and
the encoder-decoder GNN architecture.

Extracted from graph_nn.py -- no logic changes, only relocation.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv


def resolve_checkpoint_dir(config: dict) -> str:
    """
    Resolve the checkpoint directory to use for a given config, adding an
    ablation-arm subfolder whenever an ablation flag deviates from its
    default. Both training (gnn_trainer.GNN_prediction.train_validate) and
    inference (multi_task_graph_router.infer_single_query) call this with
    the SAME config, so they always resolve to the SAME path for a given
    set of ablation settings -- this is what prevents the
    "RuntimeError: size mismatch...copying a param with shape [...]"
    failure, which happens when training resolves one EncoderDecoderNet
    architecture (e.g. edge_aware=True, query_feature_dim including
    node_metadata) but a *different* config (different ablation flags) is
    later used at inference against that same model_path -- the model
    that gets constructed to load INTO has different parameter shapes
    than the checkpoint that was saved.

    Default config (ablation_disable_node_metadata=False,
    ablation_edge_aware=True) resolves to config['model_path'] UNCHANGED --
    so this is fully backward compatible with checkpoints trained before
    the ablation flags existed; no config or file-layout changes needed
    for the non-ablation (default) case.

    Non-default flags each add a suffix component, e.g.:
        ablation_disable_node_metadata=True                -> <model_path>/no_metadata/
        ablation_edge_aware=False                           -> <model_path>/no_edge_aware/
        both                                                -> <model_path>/no_metadata_no_edge_aware/
    """
    parts = []
    if config.get('ablation_disable_node_metadata', False):
        parts.append('no_metadata')
    if not config.get('ablation_edge_aware', True):
        parts.append('no_edge_aware')
    suffix = '_'.join(parts)
    return os.path.join(config['model_path'], suffix) if suffix else config['model_path']


def experiment_name(config: dict) -> str:
    """
    Human-readable label for the currently active experiment configuration
    -- used only for results-CSV row labeling (see
    GNN_prediction._save_results_csv in gnn_trainer.py), never for a
    filesystem path.

    Deliberately a SEPARATE function from resolve_checkpoint_dir(), not an
    extension of it: resolve_checkpoint_dir()'s output is a path that
    existing trained checkpoints already live at, so changing what it
    encodes (e.g. adding `semantic`) would silently break the lookup path
    for every checkpoint already trained under the current scheme. This
    function is free to fold in whatever's useful for comparing runs --
    currently `semantic`, both ablation flags, and `utility_scenario` --
    without any such constraint.
    """
    parts = [
        'semantic' if config.get('semantic', False) else 'no_semantic',
        'no_metadata' if config.get('ablation_disable_node_metadata', False) else 'metadata',
        'no_edge_aware' if not config.get('ablation_edge_aware', True) else 'edge_aware',
    ]
    scenario = config.get('utility_scenario')
    if scenario:
        parts.append(scenario)
    return '+'.join(parts)


def to_bool_tensor(x, device='cuda'):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().bool()
    return torch.tensor(x, dtype=torch.bool).to(device)


class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, common_dim, reasoning_feature_dim,
                 query_mlp_hidden_dim=None):
        super(FeatureAlign, self).__init__()
        # Question feature fusion: a real 2-layer MLP instead of a single
        # linear projection. Previously query_transform was just
        # nn.Linear(query_feature_dim, common_dim) -- no non-linearity, so
        # the concat(text_embedding, node_metadata) input (built upstream in
        # prepare_data_for_GNN / feature_builder.build_node_metadata,
        # unchanged) could only be linearly recombined before hitting the
        # GNN. query_mlp_hidden_dim defaults to common_dim if not given.
        query_mlp_hidden_dim = query_mlp_hidden_dim or common_dim
        self.query_transform = nn.Sequential(
            nn.Linear(query_feature_dim, query_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(query_mlp_hidden_dim, common_dim)
        )
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim * 2)
        self.task_transform = nn.Linear(llm_feature_dim, common_dim)
        # Reasoning prototype nodes: one shared node per domain macro-category
        # (not one per query). Each node's own input is its one-hot identity
        # within that fixed category set (see
        # feature_builder.build_reasoning_node_features), transformed into
        # the same common_dim*2 space as Question/Model nodes so all three
        # node types can sit in one flat node index space for GeneralConv.
        self.reasoning_transform = nn.Linear(reasoning_feature_dim, common_dim * 2)

    def forward(self, task_id, query_features, llm_features, reasoning_features):
        aligned_task_features = self.task_transform(task_id)
        aligned_query_features = self.query_transform(query_features)
        aligned_two_features = torch.cat([aligned_task_features, aligned_query_features], 1)
        aligned_llm_features = self.llm_transform(llm_features)
        aligned_reasoning_features = self.reasoning_transform(reasoning_features)
        # Node order MUST match graph_data.form_data.formulation's index
        # layout: Question nodes, then Reasoning prototype nodes, then Model
        # (LLM) nodes.
        aligned_features = torch.cat(
            [aligned_two_features, aligned_reasoning_features, aligned_llm_features], 0
        )
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features, in_edges,
                 reasoning_feature_dim, dropout=0.3, query_mlp_hidden_dim=None, edge_aware=True):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        # Edge-aware Message Passing ablation switch. When True (default,
        # original behavior): GeneralConv receives in_edge_channels=in_edges
        # and edge_attr is built from the Phase-4 edge feature vector
        # (Correct/Cost/Latency/.../Completion_Status) and passed into both
        # conv layers every forward() call. When False: the conv layers are
        # constructed with NO in_edge_channels (GeneralConv's default --
        # plain node-feature message passing) and edge_attr is never built
        # or passed, so the GNN has zero visibility into per-edge outcome
        # features during message passing -- only through graph structure
        # and node features. The final edge_predictor head is identical
        # either way; only the encoder path differs.
        # NOTE: this changes GeneralConv's learned parameter shapes, so a
        # checkpoint trained with one setting cannot be loaded with the
        # other -- use a distinct config['model_path'] per ablation arm.
        self.edge_aware = edge_aware
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features,
                                        reasoning_feature_dim, query_mlp_hidden_dim=query_mlp_hidden_dim)
        if self.edge_aware:
            self.encoder_conv_1 = GeneralConv(in_channels=hidden_features * 2, out_channels=hidden_features * 2,
                                              in_edge_channels=in_edges)
            self.encoder_conv_2 = GeneralConv(in_channels=hidden_features * 2, out_channels=hidden_features * 2,
                                              in_edge_channels=in_edges)
            self.edge_mlp = nn.Linear(in_edges, in_edges)
        else:
            self.encoder_conv_1 = GeneralConv(in_channels=hidden_features * 2, out_channels=hidden_features * 2)
            self.encoder_conv_2 = GeneralConv(in_channels=hidden_features * 2, out_channels=hidden_features * 2)
            self.edge_mlp = None
        # LayerNorm instead of BatchNorm1d: normalizes across feature dims per
        # sample rather than across the batch, which matters a lot on tiny
        # datasets (~35-50 training queries) where BatchNorm's running
        # mean/variance stats are noisy during training and cause a
        # significant shift once model.eval() switches to those stats.
        self.ln1 = nn.LayerNorm(hidden_features * 2)
        self.ln2 = nn.LayerNorm(hidden_features * 2)
        # Regularization: re-added here -- absent in the restructured version
        # this file was rebuilt from. At 53-68 training queries the conv
        # layers otherwise have full, unconstrained capacity. 0.3 is a
        # starting point, not tuned; sweep 0.1-0.5 against val_loss if you
        # want to find a better value. Inert automatically at inference:
        # nn.Dropout is a no-op once model.eval() is called, which
        # infer_single_query() already does.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Expressive edge scoring head: replaces the old
        # (x_ini * x).mean(dim=-1) unweighted-average predictor. That
        # collapsed the full hidden_features*2 interaction into a single
        # scalar via a plain mean, discarding any learned notion of which
        # hidden dimensions actually matter for utility prediction. Instead,
        # concatenate the initial and GNN-encoded node representations and
        # let a small MLP learn the interaction.
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_features * 4, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, task_id, query_features, llm_features, reasoning_features, edge_index,
                reasoning_edge_index, edge_mask=None, edge_can_see=None, edge_weight=None):

        if edge_mask is not None:
            edge_index_mask = edge_index[:, edge_can_see]
            edge_index_predict = edge_index[:, edge_mask]
            if edge_weight is not None:
                edge_weight_mask = edge_weight[edge_can_see]

        x_ini = (self.model_align(task_id, query_features, llm_features, reasoning_features))

        # Question<->Reasoning edges are structural (message-passing only,
        # never a prediction target) and always fully visible regardless of
        # train/val/test masking. In the edge-aware variant they carry no
        # learned edge feature, so are padded with zeros to match
        # in_edge_channels; in the non-edge-aware variant no edge_attr is
        # built at all, so no padding is needed either.
        conv_edge_index = torch.cat([edge_index_mask, reasoning_edge_index], dim=1)

        if self.edge_aware:
            edge_weight_mask = F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
            edge_weight_mask = edge_weight_mask.reshape(-1, self.in_edges)
            reasoning_edge_attr = torch.zeros(
                reasoning_edge_index.size(1), self.in_edges,
                dtype=edge_weight_mask.dtype, device=edge_weight_mask.device
            )
            conv_edge_attr = torch.cat([edge_weight_mask, reasoning_edge_attr], dim=0)

            x = F.relu(self.ln1(self.encoder_conv_1(x_ini, conv_edge_index, edge_attr=conv_edge_attr)))
            x = self.dropout1(x)
            x = self.ln2(self.encoder_conv_2(x, conv_edge_index, edge_attr=conv_edge_attr))
            x = self.dropout2(x)
        else:
            # Plain node-feature message passing -- no edge_attr built or
            # passed, so the Phase-4 edge features never reach the conv
            # layers (edge_weight/edge_weight_mask above are unused here).
            x = F.relu(self.ln1(self.encoder_conv_1(x_ini, conv_edge_index)))
            x = self.dropout1(x)
            x = self.ln2(self.encoder_conv_2(x, conv_edge_index))
            x = self.dropout2(x)

        u = x_ini[edge_index_predict[0]]
        v = x[edge_index_predict[1]]
        edge_repr = torch.cat([u, v], dim=-1)
        edge_predict = torch.sigmoid(self.edge_predictor(edge_repr).squeeze(-1))
        return edge_predict