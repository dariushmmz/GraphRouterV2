"""
Matplotlib diagnostics for GNN training runs.

Extracted from GNN_prediction in graph_nn.py -- no logic changes, only
relocation. Stays a mixin because every plot method reads self.history
(populated epoch-by-epoch during train_validate) and self.config
(for the output path) -- pulling those out to pure functions would mean
passing the entire history dict + config into each call for no benefit,
since these are only ever called from inside train_validate().
"""

import os

import matplotlib.pyplot as plt
import numpy as np


class GNNPlottingMixin:
    """
    Provides the _plot_* diagnostic methods used by GNN_prediction.train_validate().
    All methods read from self.history / self.config and write PNGs under
    self.config['model_path'].
    """

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
