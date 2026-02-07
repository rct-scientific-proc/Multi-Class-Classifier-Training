"""Metrics computation utilities."""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
from typing import Any, Dict, List, Optional, Tuple


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        score_metric: str = "accuracy",
        additional_metrics: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.score_metric = score_metric
        self.additional_metrics = additional_metrics or []
        
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated predictions and labels."""
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
        self.all_probs: List[np.ndarray] = []
        self.total_loss: float = 0.0
        self.num_batches: int = 0
    
    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ) -> None:
        """Update metrics with batch results."""
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
        
        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1
    
    def compute(self) -> Dict[str, Any]:
        """Compute all metrics."""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        metrics = {}
        
        # Average loss
        if self.num_batches > 0:
            metrics["loss"] = self.total_loss / self.num_batches
        
        # Accuracy
        metrics["accuracy"] = accuracy_score(labels, preds)
        
        # Balanced accuracy
        metrics["balanced_accuracy"] = balanced_accuracy_score(labels, preds)
        
        # F1 scores
        metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(labels, preds, average="weighted", zero_division=0)
        
        # Precision and recall
        metrics["precision"] = precision_score(labels, preds, average="macro", zero_division=0)
        metrics["recall"] = recall_score(labels, preds, average="macro", zero_division=0)
        
        # Matthews Correlation Coefficient
        metrics["mcc"] = matthews_corrcoef(labels, preds)
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
        
        # Per-class accuracy
        cm = confusion_matrix(labels, preds)
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        metrics["per_class_accuracy"] = {
            self.class_names[i]: float(acc) for i, acc in enumerate(per_class_acc)
        }
        
        # ROC AUC (if we have probabilities)
        if len(self.all_probs) > 0 and self.num_classes > 1:
            probs = np.array(self.all_probs)
            try:
                if self.num_classes == 2:
                    metrics["auc_roc"] = roc_auc_score(labels, probs[:, 1])
                else:
                    metrics["auc_roc"] = roc_auc_score(
                        labels, probs, multi_class="ovr", average="macro"
                    )
            except ValueError:
                # Can happen if some classes are not present
                metrics["auc_roc"] = 0.0
        
        return metrics
    
    def get_score(self, metrics: Optional[Dict[str, Any]] = None) -> float:
        """Get the primary score metric value."""
        if metrics is None:
            metrics = self.compute()
        
        metric_name = self.score_metric.lower()
        
        # Handle aliases
        if metric_name == "f1_score":
            metric_name = "f1_macro"
        
        if metric_name in metrics:
            return metrics[metric_name]
        else:
            raise ValueError(f"Unknown score metric: {self.score_metric}")
    
    def compute_roc_curves(self) -> Optional[Dict[str, Dict[str, List[float]]]]:
        """Compute ROC curves for each class."""
        if len(self.all_probs) == 0:
            return None
        
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        
        roc_curves = {}
        for i, class_name in enumerate(self.class_names):
            binary_labels = (labels == i).astype(int)
            class_probs = probs[:, i]
            
            try:
                fpr, tpr, thresholds = roc_curve(binary_labels, class_probs)
                roc_curves[class_name] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist()
                }
            except ValueError:
                continue
        
        return roc_curves


def save_metrics(
    metrics: Dict[str, Any],
    save_path: str,
    epoch: Optional[int] = None
) -> None:
    """Save metrics to JSON file."""
    save_dict = metrics.copy()
    if epoch is not None:
        save_dict["epoch"] = epoch
    
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2)


def save_confusion_matrix(
    confusion_mat: List[List[int]],
    class_names: List[str],
    save_path: str
) -> None:
    """Save confusion matrix to JSON file."""
    save_dict = {
        "confusion_matrix": confusion_mat,
        "class_names": class_names,
        "row_labels": "true_labels",
        "col_labels": "predicted_labels"
    }
    
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2)


def save_classification_report(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    save_path: str
) -> None:
    """Save sklearn classification report to file."""
    report = classification_report(labels, preds, target_names=class_names)
    
    with open(save_path, 'w') as f:
        f.write(report)
