import os
import shutil
import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

try:
    from router_evaluation import build_oof_predictions_df, evaluate_router_performance
except ImportError:
    build_oof_predictions_df = evaluate_router_performance = None


def to_bool_tensor(x, device='cuda'):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().bool()
    return torch.tensor(x, dtype=torch.bool).to(device)


class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, common_dim):
        super(FeatureAlign, self).__init__()
        self.query_transform = nn.Linear(query_feature_dim, common_dim)
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim * 2)
        self.task_transform = nn.Linear(llm_feature_dim, common_dim)

    def forward(self, task_id, query_features, llm_features):
        aligned_task_features = self.task_transform(task_id)
        aligned_query_features = self.query_transform(query_features)
        aligned_two_features = torch.cat([aligned_task_features, aligned_query_features], 1)
        aligned_llm_features = self.llm_transform(llm_features)
        aligned_features = torch.cat([aligned_two_features, aligned_llm_features], 0)
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features, in_edges, dropout=0.3):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features * 2, out_channels=hidden_features * 2,
                                          in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features * 2, out_channels=hidden_features * 2,
                                          in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges)
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

    def forward(self, task_id, query_features, llm_features, edge_index, edge_mask=None,
                edge_can_see=None, edge_weight=None):

        if edge_mask is not None:
            edge_index_mask = edge_index[:, edge_can_see]
            edge_index_predict = edge_index[:, edge_mask]
            if edge_weight is not None:
                edge_weight_mask = edge_weight[edge_can_see]

        edge_weight_mask = F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1, self.in_edges)
        x_ini = (self.model_align(task_id, query_features, llm_features))
        x = F.relu(self.ln1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.dropout1(x)
        x = self.ln2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        x = self.dropout2(x)

        u = x_ini[edge_index_predict[0]]
        v = x[edge_index_predict[1]]
        edge_repr = torch.cat([u, v], dim=-1)
        edge_predict = torch.sigmoid(self.edge_predictor(edge_repr).squeeze(-1))
        return edge_predict


class form_data:

    def __init__(self, device):
        self.device = device

    def formulation(self, task_id, query_feature, llm_feature, org_node, des_node, edge_feature, label, edge_mask,
                    combined_edge, train_mask, valide_mask, test_mask):
        query_features = torch.tensor(query_feature, dtype=torch.float).to(self.device)
        llm_features = torch.tensor(llm_feature, dtype=torch.float).to(self.device)
        task_id = torch.tensor(task_id, dtype=torch.float).to(self.device)

        query_indices = list(range(len(query_features)))
        llm_indices = [i + len(query_indices) for i in range(len(llm_features))]
        des_node = [(i + 1 + org_node[-1]) for i in des_node]

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
        data = Data(task_id=task_id, query_features=query_features, llm_features=llm_features, edge_index=edge_index,
                    edge_attr=edge_weight, query_indices=query_indices, llm_indices=llm_indices,
                    label=torch.tensor(label, dtype=torch.float).to(self.device),
                    edge_mask=edge_mask, combined_edge=combined_edge,
                    train_mask=train_mask, valide_mask=valide_mask, test_mask=test_mask)

        return data


class GNN_prediction:
    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features_size,
                 in_edges_size, config, device, wandb=None):

        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim,
                                        llm_feature_dim=llm_feature_dim,
                                        hidden_features=hidden_features_size,
                                        in_edges=in_edges_size,
                                        dropout=config.get('dropout', 0.3)).to(device)
        self.device = device
        self.config = config

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        # Cosine-anneal the LR to (near) zero over training instead of holding
        # it static -- on a small dataset a static LR tends to keep bouncing
        # around local minima late in training rather than settling.
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config['train_epoch'], eta_min=1e-6
        )
        self.criterion = torch.nn.BCELoss()

        # Metric tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'test_predict': [],
            'test_golden': [],
        }

    # ----------------- Helpers -----------------
    @staticmethod
    def _to_numpy(x):
        return x.detach().cpu().item() if torch.is_tensor(x) else float(x)

    def _print_epoch_summary(self, epoch):
        print(
            f"[Epoch {epoch + 1:4d}/{self.config['train_epoch']}] "
            f"train_loss={self.history['train_loss'][-1]:.4f} | "
            f"val_loss={self.history['val_loss'][-1]:.4f} | "
            f"test_loss={self.history['test_loss'][-1]:.4f} | "
            f"val_acc={self.history['val_accuracy'][-1]:.4f} | "
            f"val_f1={self.history['val_f1'][-1]:.4f} | "
            f"test_pred={self.history['test_predict'][-1]:.4f} | "
            f"test_golden={self.history['test_golden'][-1]:.4f}"
        )

    # ----------------- Plotting -----------------
    def _plot_losses(self, epochs):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, self.history['train_loss'], marker='o', markersize=3,
                linewidth=1.6, label='Train Loss', color='#1f77b4')
        ax.plot(epochs, self.history['val_loss'], marker='s', markersize=3,
                linewidth=1.6, label='Validation Loss', color='#ff7f0e')
        ax.plot(epochs, self.history['test_loss'], marker='^', markersize=3,
                linewidth=1.6, label='Test Loss', color='#2ca02c')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('BCE Loss', fontsize=12)
        ax.set_title('Training / Validation / Test Loss', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', frameon=True, fontsize=11)
        fig.tight_layout()
        path = os.path.join(self.config['model_path'], 'losses.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved loss curve -> {path}")

    def _plot_metrics(self, epochs):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, self.history['val_accuracy'], marker='o', markersize=3,
                linewidth=1.6, label='Validation Accuracy', color='#9467bd')
        ax.plot(epochs, self.history['val_f1'], marker='s', markersize=3,
                linewidth=1.6, label='Validation Macro-F1', color='#d62728')
        best_idx = int(np.argmax(self.history['val_f1']))
        ax.axvline(epochs[best_idx], color='gray', linestyle=':', alpha=0.7,
                   label=f"Best F1 @ epoch {epochs[best_idx]}")
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Validation Accuracy & Macro-F1', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', frameon=True, fontsize=11)
        fig.tight_layout()
        path = os.path.join(self.config['model_path'], 'metrics.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved metrics curve -> {path}")

    def _plot_test_results(self, epochs):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, self.history['test_predict'], marker='o', markersize=3,
                linewidth=1.6, label='Predicted (test)', color='#1f77b4')
        ax.plot(epochs, self.history['test_golden'], marker='^', markersize=3,
                linewidth=1.6, linestyle='--', label='Golden (test)', color='#ff7f0e')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_title('Test: Predicted vs. Golden', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', frameon=True, fontsize=11)
        fig.tight_layout()
        path = os.path.join(self.config['model_path'], 'test_results.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved test results -> {path}")

    def _plot_best_bar(self):
        """Bar chart comparing predicted vs golden at the best-F1 epoch."""
        best_idx = int(np.argmax(self.history['val_f1']))
        pred = self.history['test_predict'][best_idx]
        gold = self.history['test_golden'][best_idx]

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(['Predicted', 'Golden'], [pred, gold],
                      color=['#1f77b4', '#ff7f0e'], width=0.55, edgecolor='black')
        for b in bars:
            ax.annotate(f"{b.get_height():.4f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_title(f'Best-F1 Epoch Test Result\n(epoch {best_idx + 1})',
                     fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, max(pred, gold) * 1.2)
        fig.tight_layout()
        path = os.path.join(self.config['model_path'], 'best_epoch_comparison.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved best-epoch bar chart -> {path}")

    def _plot_dashboard(self, epochs):
        """Single figure with all curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], label='Train', color='#1f77b4')
        ax.plot(epochs, self.history['val_loss'], label='Val', color='#ff7f0e')
        ax.plot(epochs, self.history['test_loss'], label='Test', color='#2ca02c')
        ax.set_title('Loss', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
        ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

        ax = axes[0, 1]
        ax.plot(epochs, self.history['val_accuracy'], label='Val Acc', color='#9467bd')
        ax.plot(epochs, self.history['val_f1'], label='Val F1', color='#d62728')
        best_idx = int(np.argmax(self.history['val_f1']))
        ax.axvline(epochs[best_idx], color='gray', linestyle=':',
                   alpha=0.7, label=f"Best F1 @ {epochs[best_idx]}")
        ax.set_ylim(0, 1.02)
        ax.set_title('Validation Metrics', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
        ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

        ax = axes[1, 0]
        ax.plot(epochs, self.history['test_predict'], label='Predicted',
                color='#1f77b4', marker='o', markersize=3)
        ax.plot(epochs, self.history['test_golden'], label='Golden',
                color='#ff7f0e', marker='^', markersize=3, linestyle='--')
        ax.set_title('Test: Predicted vs Golden', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Value')
        ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

        ax = axes[1, 1]
        pred = self.history['test_predict'][best_idx]
        gold = self.history['test_golden'][best_idx]
        bars = ax.bar(['Predicted', 'Golden'], [pred, gold],
                      color=['#1f77b4', '#ff7f0e'], width=0.55, edgecolor='black')
        for b in bars:
            ax.annotate(f"{b.get_height():.4f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=11, fontweight='bold')
        ax.set_title(f'Best Epoch Test ({best_idx + 1})', fontweight='bold')
        ax.set_ylabel('Mean Value')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, max(pred, gold) * 1.2)

        fig.suptitle('GNN Training Dashboard', fontsize=16, fontweight='bold', y=1.00)
        fig.tight_layout()
        path = os.path.join(self.config['model_path'], 'dashboard.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved dashboard -> {path}")

    # ----------------- Main training loop -----------------
    def train_validate(self, data, data_validate, data_for_test, make_plots=True):

        best_f1 = -1
        best_val_loss_ckpt = float('inf')
        os.makedirs(self.config['model_path'], exist_ok=True)
        # BUGFIX: was "best_mode.pth" (typo) in the version this was rebuilt
        # from -- also, multi_task_graph_router.py's infer_single_query()
        # does torch.load(self.config['model_path'], ...) directly, treating
        # model_path as a FILE. Since this class now treats model_path as a
        # DIRECTORY (os.makedirs above), that call would hit
        # IsADirectoryError. Fixed on the router side to load from
        # os.path.join(config['model_path'], "best_model.pth") to match --
        # see multi_task_graph_router.py. Your config['model_path'] value
        # needs to be a directory now (e.g. "checkpoints/"), not a file path
        # like the old "checkpoints/best_model.pth".
        self.save_path = os.path.join(self.config['model_path'], "best_model.pth")

        self.num_edges = len(data.edge_attr)

        self.train_mask = to_bool_tensor(data.train_mask, self.device)
        self.valide_mask = to_bool_tensor(data.valide_mask, self.device)
        self.test_mask = to_bool_tensor(data.test_mask, self.device)

        # Early stopping on val_loss plateau. Checkpoint selection itself is
        # UNCHANGED (still best val_f1) -- this only stops wasted/potentially
        # counterproductive continued training once val_loss stops
        # improving, rather than always running the full train_epoch count.
        # Set early_stopping_patience: null (or omit) in config to disable.
        patience = self.config.get('early_stopping_patience', 20)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(self.config['train_epoch']):
            # --- Train ---
            self.model.train()
            loss_mean = 0
            mask_train = data.edge_mask
            for inter in range(self.config['batch_size']):
                mask = mask_train.clone().bool()
                random_mask = torch.rand(mask.size()) < self.config['train_mask_rate']
                random_mask = random_mask.to(torch.bool).to(self.device)
                mask = torch.where(mask & random_mask,
                                   torch.tensor(False, dtype=torch.bool, device=self.device),
                                   mask).bool()
                edge_can_see = torch.logical_and(~mask, self.train_mask)
                self.optimizer.zero_grad()
                predicted_edges = self.model(task_id=data.task_id,
                                             query_features=data.query_features,
                                             llm_features=data.llm_features,
                                             edge_index=data.edge_index,
                                             edge_mask=mask,
                                             edge_can_see=edge_can_see,
                                             edge_weight=data.combined_edge)
                loss = self.criterion(predicted_edges.reshape(-1),
                                      data.label[mask].reshape(-1))
                loss_mean += loss
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            loss_mean = loss_mean / self.config['batch_size']

            # --- Validate ---
            self.model.eval()
            mask_validate = to_bool_tensor(data_validate.edge_mask, self.device)
            edge_can_see = self.train_mask

            with torch.no_grad():
                predicted_edges_validate = self.model(
                    task_id=data_validate.task_id,
                    query_features=data_validate.query_features,
                    llm_features=data_validate.llm_features,
                    edge_index=data_validate.edge_index,
                    edge_mask=mask_validate,
                    edge_can_see=edge_can_see,
                    edge_weight=data_validate.combined_edge)

                observe_edge = predicted_edges_validate.reshape(-1, self.config['llm_num'])
                observe_idx = torch.argmax(observe_edge, dim=1)
                value_validate = data_validate.edge_attr[mask_validate].reshape(-1, self.config['llm_num'])
                label_idx = torch.argmax(value_validate, dim=1)

                correct = (observe_idx == label_idx).sum().item()
                total = label_idx.size(0)
                validate_accuracy = correct / total

                f1 = f1_score(label_idx.cpu().numpy(),
                              observe_idx.cpu().numpy(), average='macro')
                loss_validate = self.criterion(
                    predicted_edges_validate.reshape(-1),
                    data_validate.label[mask_validate].reshape(-1))

                if f1 > best_f1:
                    best_f1 = f1

                # Checkpoint on val_loss, not val_f1: with a 10% split on ~71
                # queries the validation set is only ~7 queries -- macro-F1
                # there is noisy and discrete (one query flipping shifts it by
                # 14.3%), so it tends to save whichever epoch got lucky on
                # argmax rather than the best-calibrated model. val_loss
                # (continuous, from probabilities) is a steadier signal.
                if loss_validate < best_val_loss_ckpt:
                    best_val_loss_ckpt = loss_validate
                    torch.save(self.model.state_dict(), self.save_path)

                test_result, test_loss = self.test(data_for_test, self.save_path)

            # --- Track metrics ---
            self.history['train_loss'].append(self._to_numpy(loss_mean))
            self.history['val_loss'].append(self._to_numpy(loss_validate))
            self.history['test_loss'].append(self._to_numpy(test_loss))
            self.history['val_accuracy'].append(float(validate_accuracy))
            self.history['val_f1'].append(float(f1))
            self.history['test_predict'].append(self._to_numpy(test_result))
            self.history['test_golden'].append(
                self._to_numpy(
                    data_for_test.edge_attr[to_bool_tensor(data_for_test.edge_mask, self.device)]
                    .reshape(-1, self.config['llm_num'])
                    [torch.arange(
                        data_for_test.edge_attr[to_bool_tensor(data_for_test.edge_mask, self.device)]
                        .reshape(-1, self.config['llm_num']).size(0)),
                     torch.argmax(
                         data_for_test.edge_attr[to_bool_tensor(data_for_test.edge_mask, self.device)]
                         .reshape(-1, self.config['llm_num']), dim=1)].mean()
                )
            )

            self._print_epoch_summary(epoch)

            # --- Early stopping check (after logging this epoch) ---
            if patience is not None:
                val_loss_value = self.history['val_loss'][-1]
                if val_loss_value < best_val_loss - 1e-4:
                    best_val_loss = val_loss_value
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"[early_stopping] val_loss hasn't improved for {patience} epochs "
                          f"(best={best_val_loss:.4f}) -- stopping at epoch {epoch + 1}/{self.config['train_epoch']}. "
                          f"Best checkpoint (val_f1={best_f1:.4f}) already saved to {self.save_path}.")
                    stopped_early = True
                    break

        # --- Final summary & plots ---
        print("\n========== Training Complete ==========")
        print(f"Best Validation F1 (informational only, not used for checkpointing): {best_f1:.4f}")
        print(f"Best Validation Loss (checkpoint criterion): {best_val_loss_ckpt:.4f}")
        if stopped_early:
            print(f"(stopped early at epoch {len(self.history['train_loss'])} of "
                  f"{self.config['train_epoch']} configured)")
        # BUGFIX: was hardcoded to config['train_epoch'] -- with early
        # stopping, history can be SHORTER than that, and matplotlib would
        # error on mismatched x/y lengths. Use actual recorded epoch count.
        epochs = list(range(1, len(self.history['train_loss']) + 1))

        if make_plots:
            self._plot_losses(epochs)
            self._plot_metrics(epochs)
            self._plot_test_results(epochs)
            self._plot_best_bar()
            self._plot_dashboard(epochs)

    # ----------------- Test -----------------
    def test(self, data, model_path):
        self.model.eval()
        mask = to_bool_tensor(data.edge_mask, self.device)

        edge_can_see = torch.logical_or(self.valide_mask, self.train_mask)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,
                                      query_features=data.query_features,
                                      llm_features=data.llm_features,
                                      edge_index=data.edge_index,
                                      edge_mask=mask,
                                      edge_can_see=edge_can_see,
                                      edge_weight=data.combined_edge)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(edge_predict, label)

        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        max_idx = torch.argmax(edge_predict, dim=1)
        value_test = data.edge_attr[mask].reshape(-1, self.config['llm_num'])
        label_idx = torch.argmax(value_test, dim=1)
        row_indices = torch.arange(len(value_test))
        result = value_test[row_indices, max_idx].mean()
        result_golden = value_test[row_indices, label_idx].mean()
        print(f"result_predict: {result.item():.6f} | result_golden: {result_golden.item():.6f}")

        return result, loss_test

    # ----------------- Raw per-row predictions (for router-level eval) -----------------
    def predict_raw(self, data):
        """
        Return the model's raw predicted utility for every row selected by
        data.edge_mask, in the same row order as edge_mask. Unlike test(),
        which only returns the fold's mean selected-utility, this is what
        router_evaluation.build_oof_predictions_df() needs to score Gap
        Closed / NDCG / Pareto metrics across pooled out-of-fold queries
        instead of per-fold summary stats.
        """
        self.model.eval()
        mask = to_bool_tensor(data.edge_mask, self.device)
        edge_can_see = torch.logical_or(self.valide_mask, self.train_mask)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,
                                      query_features=data.query_features,
                                      llm_features=data.llm_features,
                                      edge_index=data.edge_index,
                                      edge_mask=mask,
                                      edge_can_see=edge_can_see,
                                      edge_weight=data.combined_edge)
        return edge_predict.detach().cpu().numpy().reshape(-1)


# ----------------------------------------------------------------------------
# K-Fold Cross Validation
# ----------------------------------------------------------------------------
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