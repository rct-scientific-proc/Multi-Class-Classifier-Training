"""Plotting utilities for training visualization."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight'
    })


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    score_metric: str = "accuracy"
) -> None:
    """Plot training and validation loss/score curves."""
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Score curve
    ax2 = axes[1]
    ax2.plot(epochs, history['train_score'], 'b-', label=f'Train {score_metric}', linewidth=2)
    ax2.plot(epochs, history['val_score'], 'r-', label=f'Val {score_metric}', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(score_metric.capitalize())
    ax2.set_title(f'Training and Validation {score_metric.capitalize()}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_learning_rate(
    history: Dict[str, List[float]],
    save_path: str
) -> None:
    """Plot learning rate schedule."""
    set_plot_style()
    
    if 'learning_rate' not in history or not history['learning_rate']:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history['learning_rate']) + 1)
    
    ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(
    confusion_mat: List[List[int]],
    class_names: List[str],
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> None:
    """Plot confusion matrix as a heatmap."""
    set_plot_style()
    
    cm = np.array(confusion_mat)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    else:
        cm_normalized = cm
    
    # Adjust figure size based on number of classes
    n_classes = len(class_names)
    fig_size = max(8, n_classes * 0.8)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x-axis labels if many classes
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm_normalized.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_normalized[i, j] if normalize else cm[i, j]
            ax.text(j, i, format(value, fmt),
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=8 if n_classes > 10 else 10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_roc_curves(
    roc_curves: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title: str = "ROC Curves"
) -> None:
    """Plot ROC curves for all classes."""
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette
    n_classes = len(roc_curves)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_classes, 10)))
    if n_classes > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_classes, 20)))
    
    for idx, (class_name, curves) in enumerate(roc_curves.items()):
        fpr = curves['fpr']
        tpr = curves['tpr']
        
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {auc:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    
    # Position legend outside if many classes
    if n_classes > 6:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        ax.legend(loc='lower right')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_per_class_accuracy(
    per_class_acc: Dict[str, float],
    save_path: str,
    title: str = "Per-Class Accuracy"
) -> None:
    """Plot per-class accuracy as a bar chart."""
    set_plot_style()
    
    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    
    # Adjust figure width based on number of classes
    fig_width = max(10, len(classes) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Color bars based on accuracy
    colors = plt.cm.RdYlGn(np.array(accuracies))
    
    bars = ax.bar(classes, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    # Add mean line
    mean_acc = np.mean(accuracies)
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.3f}')
    ax.legend()
    
    # Rotate x-axis labels if many classes
    if len(classes) > 10:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_metrics_comparison(
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Metrics Summary"
) -> None:
    """Plot bar chart comparing different metrics."""
    set_plot_style()
    
    # Select relevant metrics
    metric_names = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 
                    'precision', 'recall', 'mcc']
    
    plot_metrics = {k: v for k, v in metrics.items() if k in metric_names}
    
    if not plot_metrics:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(plot_metrics.keys())
    values = list(plot_metrics.values())
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def generate_all_figures(
    output_dir: Path,
    history: Dict[str, List[float]],
    metrics: Dict[str, Any],
    class_names: List[str],
    split_name: str,
    score_metric: str = "accuracy",
    roc_curves: Optional[Dict] = None
) -> None:
    """Generate and save all figures."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Training curves (only if history is provided)
    if history and 'train_loss' in history and history['train_loss']:
        plot_training_curves(
            history, 
            str(figures_dir / "training_curves.png"),
            score_metric
        )
        plot_learning_rate(
            history,
            str(figures_dir / "learning_rate.png")
        )
    
    # Confusion matrix
    if "confusion_matrix" in metrics:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            str(figures_dir / f"{split_name}_confusion_matrix.png"),
            normalize=True,
            title=f"{split_name.capitalize()} Confusion Matrix (Normalized)"
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            str(figures_dir / f"{split_name}_confusion_matrix_counts.png"),
            normalize=False,
            title=f"{split_name.capitalize()} Confusion Matrix (Counts)"
        )
    
    # ROC curves
    if roc_curves:
        plot_roc_curves(
            roc_curves,
            str(figures_dir / f"{split_name}_roc_curves.png"),
            title=f"{split_name.capitalize()} ROC Curves"
        )
    
    # Per-class accuracy
    if "per_class_accuracy" in metrics:
        plot_per_class_accuracy(
            metrics["per_class_accuracy"],
            str(figures_dir / f"{split_name}_per_class_accuracy.png"),
            title=f"{split_name.capitalize()} Per-Class Accuracy"
        )
    
    # Metrics comparison
    plot_metrics_comparison(
        metrics,
        str(figures_dir / f"{split_name}_metrics_summary.png"),
        title=f"{split_name.capitalize()} Metrics Summary"
    )
