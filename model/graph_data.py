"""
PyTorch Geometric Data formulation for the graph router.

Extracted from graph_nn.py -- no logic changes, only relocation, PLUS the
addition of a real Reasoning node type (see formulation() docstring below).
"""

import torch
from torch_geometric.data import Data


class form_data:

    def __init__(self, device):
        self.device = device

    def formulation(self, task_id, query_feature, llm_feature, org_node, des_node, edge_feature, label, edge_mask,
                    combined_edge, train_mask, valide_mask, test_mask, reasoning_feature, reasoning_node_id):
        """
        Node index layout (flat, shared across all three node types):
            [0, num_query)                                  Question nodes
            [num_query, num_query + num_reasoning)           Reasoning prototype nodes
            [num_query + num_reasoning, ... + num_llms)      Model (LLM) nodes

        reasoning_feature: (num_reasoning, num_reasoning) identity matrix from
            feature_builder.build_reasoning_node_features -- one shared
            prototype node per domain macro-category, NOT one per query.
        reasoning_node_id: (num_query,) int array from
            feature_builder.assign_reasoning_node_index -- which prototype
            node (by index into reasoning_feature's rows) each query belongs
            to.

        org_node/des_node/edge_feature/edge_mask/combined_edge/label
        continue to describe ONLY the Question->Model prediction edges,
        exactly as before -- the new Question<->Reasoning edges are
        structural (message-passing only) and are kept in a separate
        `reasoning_edge_index` tensor so none of the existing
        train/val/test masking logic for the prediction task changes.
        """
        query_features = torch.tensor(query_feature, dtype=torch.float).to(self.device)
        llm_features = torch.tensor(llm_feature, dtype=torch.float).to(self.device)
        task_id = torch.tensor(task_id, dtype=torch.float).to(self.device)
        reasoning_features = torch.tensor(reasoning_feature, dtype=torch.float).to(self.device)

        num_query = len(query_features)
        num_reasoning = reasoning_features.shape[0]

        query_indices = list(range(num_query))
        reasoning_indices = [num_query + i for i in range(num_reasoning)]
        llm_indices = [num_query + num_reasoning + i for i in range(len(llm_features))]

        # BUGFIX vs. the old two-node-type version: des_node used to be
        # offset by `org_node[-1] + 1` (i.e. straight past the Question
        # nodes) to land in LLM-node space. With Reasoning nodes now sitting
        # between Question and LLM nodes in the flat index space, the offset
        # must also skip past num_reasoning, or Question->Model edges would
        # land on/inside the Reasoning node block instead of the LLM block.
        des_node = [(i + num_query + num_reasoning) for i in des_node]

        edge_index = torch.tensor([org_node, des_node], dtype=torch.long).to(self.device)
        edge_weight = torch.tensor(edge_feature, dtype=torch.float).reshape(-1, 1).to(self.device)
        # NOTE: width is now dynamic (Phase 4 edge feature vector:
        # Correct, Cost_norm, Latency_norm, Input_Tokens_norm, Output_Tokens_norm,
        # Completion_Status_onehot, Error_Type_onehot) instead of the old
        # fixed-width [cost, effect] pair. Do not hardcode reshape(-1, 2) here.
        combined_edge = torch.as_tensor(combined_edge, dtype=torch.float)
        combined_edge = combined_edge.reshape(len(org_node) if combined_edge.dim() == 1 else combined_edge.shape[0],
                                              -1).to(self.device)

        combined_edge = torch.cat((edge_weight, combined_edge), dim=-1)

        # Question<->Reasoning structural edges: bidirectional so message
        # passing flows both ways across the two GeneralConv layers. One
        # query maps to exactly one prototype node (its domain
        # macro-category); many queries can, and typically do, share the
        # same prototype node.
        reasoning_node_id = list(reasoning_node_id)
        q_side = [query_indices[i] for i in range(num_query)]
        r_side = [reasoning_indices[reasoning_node_id[i]] for i in range(num_query)]
        reasoning_edge_index = torch.tensor(
            [q_side + r_side, r_side + q_side], dtype=torch.long
        ).to(self.device)

        data = Data(task_id=task_id, query_features=query_features, llm_features=llm_features,
                    reasoning_features=reasoning_features, edge_index=edge_index,
                    reasoning_edge_index=reasoning_edge_index,
                    edge_attr=edge_weight, query_indices=query_indices, llm_indices=llm_indices,
                    reasoning_indices=reasoning_indices,
                    label=torch.tensor(label, dtype=torch.float).to(self.device),
                    edge_mask=edge_mask, combined_edge=combined_edge,
                    train_mask=train_mask, valide_mask=valide_mask, test_mask=test_mask)

        return data
