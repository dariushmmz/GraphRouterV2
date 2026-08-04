"""
GNN_prediction: the training loop, validation, and test/inference-scoring
class for the encoder-decoder graph router model.

Extracted from graph_nn.py -- no logic changes, only relocation. Plotting
methods moved to gnn_plotting.py (GNNPlottingMixin); NN modules moved to
graph_layers.py.
"""

import os
from datetime import datetime

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from model.graph_layers import EncoderDecoderNet, to_bool_tensor, resolve_checkpoint_dir, experiment_name
from model.gnn_plotting import GNNPlottingMixin


class GNN_prediction(GNNPlottingMixin):
    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features_size,
                 in_edges_size, reasoning_feature_dim, config, device, wandb=None):

        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim,
                                        llm_feature_dim=llm_feature_dim,
                                        hidden_features=hidden_features_size,
                                        in_edges=in_edges_size,
                                        reasoning_feature_dim=reasoning_feature_dim,
                                        dropout=config.get('dropout', 0.3),
                                        edge_aware=config.get('ablation_edge_aware', True)).to(device)
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

    def _save_results_csv(self, best_f1, best_val_loss_ckpt, stopped_early, fold=None):
        """
        Append this run's final summary metrics as one row to a shared
        results CSV, labeled with the active experiment configuration
        (see experiment_name() in graph_layers.py -- semantic + both
        ablation flags + utility_scenario). Creates the file with a header
        on first write; appends on every subsequent call, so results from
        every ablation arm you've run land in one comparable table instead
        of scattered console logs.

        fold: pass the fold index when called from k-fold CV (e.g.
        kfold_cv.py should pass fold=<fold_idx> into train_validate(),
        which forwards it here) so each fold's row is distinguishable.
        Leave None for a single non-k-fold run.

        After appending, recomputes an 'is_best_fold' column: within each
        experiment_name group that HAS fold values (i.e. actual k-fold
        runs -- single non-k-fold rows are excluded from this), the row
        with the lowest best_val_loss is flagged True and all others in
        that group False. Uses best_val_loss (not best_val_f1) since
        that's the same criterion already used for per-epoch checkpoint
        selection elsewhere in this class -- keeps "best" consistent
        across the codebase. Recomputed (not just set once) on every call,
        since which fold is "best so far" can change as later folds finish.

        Path is config['ablation_results_csv'] if set, else
        'checkpoints/ablation_results.csv'. Runs on every train_validate()
        call regardless of make_plots, so it's captured even if plotting
        is skipped or fails.
        """
        csv_path = self.config.get('ablation_results_csv', 'checkpoints/ablation_results.csv')
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        row = {
            'experiment_name': experiment_name(self.config),
            'fold': fold if fold is not None else '',
            'semantic': self.config.get('semantic', False),
            'ablation_disable_node_metadata': self.config.get('ablation_disable_node_metadata', False),
            'ablation_edge_aware': self.config.get('ablation_edge_aware', True),
            'utility_scenario': self.config.get('utility_scenario', ''),
            'checkpoint_path': self.save_path,
            'epochs_run': len(self.history['train_loss']),
            'configured_epochs': self.config['train_epoch'],
            'stopped_early': stopped_early,
            'best_val_f1': best_f1,
            'best_val_loss': best_val_loss_ckpt,
            'final_val_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else float('nan'),
            'final_test_predict': self.history['test_predict'][-1] if self.history['test_predict'] else float('nan'),
            'final_test_golden': self.history['test_golden'][-1] if self.history['test_golden'] else float('nan'),
            'is_best_fold': False,  # recomputed below across the whole file
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }

        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])

        df['is_best_fold'] = False
        has_fold = df['fold'].astype(str).str.strip() != ''
        for _, group in df[has_fold].groupby('experiment_name'):
            best_idx = group['best_val_loss'].astype(float).idxmin()
            df.loc[best_idx, 'is_best_fold'] = True

        df.to_csv(csv_path, index=False)
        fold_suffix = f" fold={fold}" if fold is not None else ""
        print(f"[results_csv] appended '{row['experiment_name']}'{fold_suffix} -> '{csv_path}'")

    # ----------------- Main training loop -----------------
    def train_validate(self, data, data_validate, data_for_test, make_plots=True, fold=None):

        best_f1 = -1
        best_val_loss_ckpt = float('inf')
        # Ablation-aware checkpoint dir: resolves to config['model_path']
        # unchanged when both ablation flags are at their defaults (fully
        # backward compatible); otherwise adds a subfolder per active
        # ablation flag so different arms' checkpoints never collide under
        # the same path -- see resolve_checkpoint_dir() in graph_layers.py.
        checkpoint_dir = resolve_checkpoint_dir(self.config)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[checkpoint] ablation settings (ablation_disable_node_metadata="
              f"{self.config.get('ablation_disable_node_metadata', False)}, "
              f"ablation_edge_aware={self.config.get('ablation_edge_aware', True)}) "
              f"resolve to checkpoint dir: {checkpoint_dir}")
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
        self.save_path = os.path.join(checkpoint_dir, "best_model.pth")

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
                                             reasoning_features=data.reasoning_features,
                                             edge_index=data.edge_index,
                                             reasoning_edge_index=data.reasoning_edge_index,
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
                    reasoning_features=data_validate.reasoning_features,
                    edge_index=data_validate.edge_index,
                    reasoning_edge_index=data_validate.reasoning_edge_index,
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

        self._save_results_csv(best_f1=best_f1, best_val_loss_ckpt=self._to_numpy(best_val_loss_ckpt),
                               stopped_early=stopped_early, fold=fold)

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
                                      reasoning_features=data.reasoning_features,
                                      edge_index=data.edge_index,
                                      reasoning_edge_index=data.reasoning_edge_index,
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
                                      reasoning_features=data.reasoning_features,
                                      edge_index=data.edge_index,
                                      reasoning_edge_index=data.reasoning_edge_index,
                                      edge_mask=mask,
                                      edge_can_see=edge_can_see,
                                      edge_weight=data.combined_edge)
        return edge_predict.detach().cpu().numpy().reshape(-1)