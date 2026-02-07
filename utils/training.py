"""Training utilities and loss functions."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad, Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
    SequentialLR,
    _LRScheduler
)
from typing import List, Optional, Tuple, Union

from .config import Config


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Compute loss
        if self.weight is not None:
            log_probs = log_probs * self.weight.unsqueeze(0)
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_loss_function(
    config: Config,
    class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """Create loss function based on configuration."""
    loss_name = config.loss_function.lower()
    
    if loss_name == "crossentropyloss":
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_name == "labelsmoothingcrossentropy":
        return LabelSmoothingCrossEntropy(
            smoothing=config.label_smoothing,
            weight=class_weights
        )
    
    elif loss_name == "focalloss":
        focal_config = config.focal_loss
        return FocalLoss(
            alpha=focal_config.get("alpha", 1.0),
            gamma=focal_config.get("gamma", 2.0),
            weight=class_weights
        )
    
    elif loss_name == "weightedcrossentropyloss":
        if class_weights is None:
            raise ValueError("WeightedCrossEntropyLoss requires class weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    
    else:
        raise ValueError(f"Unknown loss function: {config.loss_function}")


def create_optimizer(config: Config, model: nn.Module) -> Optimizer:
    """Create optimizer based on configuration."""
    optimizer_name = config.optimizer.lower()
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name == "adam":
        adam_config = config.adam
        return Adam(
            params,
            lr=config.learning_rate,
            betas=tuple(adam_config.get("betas", [0.9, 0.999])),
            eps=adam_config.get("eps", 1e-8),
            weight_decay=config.weight_decay,
            amsgrad=adam_config.get("amsgrad", False)
        )
    
    elif optimizer_name == "adamw":
        adam_config = config.adam
        return AdamW(
            params,
            lr=config.learning_rate,
            betas=tuple(adam_config.get("betas", [0.9, 0.999])),
            eps=adam_config.get("eps", 1e-8),
            weight_decay=config.weight_decay,
            amsgrad=adam_config.get("amsgrad", False)
        )
    
    elif optimizer_name == "sgd":
        sgd_config = config.sgd
        return SGD(
            params,
            lr=config.learning_rate,
            momentum=sgd_config.get("momentum", 0.9),
            weight_decay=config.weight_decay,
            nesterov=sgd_config.get("nesterov", True)
        )
    
    elif optimizer_name == "rmsprop":
        return RMSprop(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    elif optimizer_name == "adagrad":
        return Adagrad(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(
    config: Config,
    optimizer: Optimizer,
    steps_per_epoch: int
) -> Tuple[Optional[_LRScheduler], bool]:
    """
    Create learning rate scheduler based on configuration.
    
    Returns:
        Tuple of (scheduler, is_batch_scheduler) where is_batch_scheduler
        indicates if scheduler should step per batch (True) or per epoch (False).
    """
    scheduler_name = config.learning_rate_scheduler.lower()
    is_batch_scheduler = False
    
    # Base scheduler
    if scheduler_name == "steplr":
        step_config = config.step_lr
        scheduler = StepLR(
            optimizer,
            step_size=step_config.get("step_size", 5),
            gamma=step_config.get("gamma", 0.1)
        )
    
    elif scheduler_name == "multisteplr":
        multi_config = config.multi_step_lr
        scheduler = MultiStepLR(
            optimizer,
            milestones=multi_config.get("milestones", [30, 60, 90]),
            gamma=multi_config.get("gamma", 0.1)
        )
    
    elif scheduler_name == "exponentiallr":
        exp_config = config.exponential_lr
        scheduler = ExponentialLR(
            optimizer,
            gamma=exp_config.get("gamma", 0.95)
        )
    
    elif scheduler_name == "cosineannealinglr":
        cosine_config = config.cosine_annealing_lr
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_config.get("T_max", config.num_epochs),
            eta_min=cosine_config.get("eta_min", 1e-6)
        )
    
    elif scheduler_name == "cosineannealingwarmrestarts":
        cosine_warm_config = config.cosine_annealing_warm_restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_warm_config.get("T_0", 10),
            T_mult=cosine_warm_config.get("T_mult", 2),
            eta_min=cosine_warm_config.get("eta_min", 1e-6)
        )
    
    elif scheduler_name == "reducelronplateau":
        plateau_config = config.reduce_lr_on_plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=plateau_config.get("mode", "max"),
            factor=plateau_config.get("factor", 0.1),
            patience=plateau_config.get("patience", 5),
            threshold=plateau_config.get("threshold", 0.0001),
            min_lr=plateau_config.get("min_lr", 1e-7)
        )
    
    elif scheduler_name == "onecyclelr":
        one_cycle_config = config.one_cycle_lr
        scheduler = OneCycleLR(
            optimizer,
            max_lr=one_cycle_config.get("max_lr", 0.01),
            epochs=config.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=one_cycle_config.get("pct_start", 0.3),
            anneal_strategy=one_cycle_config.get("anneal_strategy", "cos"),
            div_factor=one_cycle_config.get("div_factor", 25.0),
            final_div_factor=one_cycle_config.get("final_div_factor", 10000.0)
        )
        is_batch_scheduler = True
    
    elif scheduler_name in ("none", "null", ""):
        return None, False
    
    else:
        raise ValueError(f"Unknown scheduler: {config.learning_rate_scheduler}")
    
    # Add warmup if specified
    if config.warmup_epochs > 0 and not is_batch_scheduler:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=config.warmup_start_factor,
            total_iters=config.warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[config.warmup_epochs]
        )
    
    return scheduler, is_batch_scheduler


class EarlyStopping:
    """Early stopping to stop training when validation score doesn't improve."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 3,
        mode: str = "max"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.checkpoints: List[Tuple[float, str]] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        score: float,
        config: Config,
        class_names: List[str]
    ) -> str:
        """Save a checkpoint and manage top-k."""
        checkpoint_path = str(self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth")
        
        # Handle DataParallel
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "score": score,
            "config": config.__dict__,
            "class_names": class_names
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Add to list and sort
        self.checkpoints.append((score, checkpoint_path))
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.keep_top_k:
            _, old_path = self.checkpoints.pop()
            if Path(old_path).exists():
                Path(old_path).unlink()
        
        return checkpoint_path
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        if self.checkpoints:
            return self.checkpoints[0][1]
        return None


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: torch.device = torch.device("cpu")
) -> Tuple[int, float]:
    """Load a checkpoint and return (epoch, score)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint.get("epoch", 0), checkpoint.get("score", 0.0)
