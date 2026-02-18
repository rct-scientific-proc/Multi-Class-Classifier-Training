"""
Multi-class image classifier training script.

Usage:
    python train.py --config example_config.yaml

This script trains a multi-class image classifier using torchvision models,
with support for various optimizers, schedulers, loss functions, and
data augmentation techniques.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from utils.config import Config, load_config, save_config
from utils.data import create_data_loaders, compute_class_weights
from utils.metrics import MetricsTracker, save_metrics, save_confusion_matrix, save_classification_report
from utils.models import create_model, export_to_onnx, export_to_torchscript
from utils.network_viz import generate_all_network_figures
from utils.plotting import generate_all_figures, plot_stratification_histograms
from utils.training import (
    create_loss_function,
    create_optimizer,
    create_scheduler,
    EarlyStopping,
    CheckpointManager,
    load_checkpoint
)


def setup_logging(log_level: str, output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: Config) -> torch.device:
    """Get the device to use for training."""
    device_str = config.device.lower()
    
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_str.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(device_str)
    else:
        return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
    scaler: Optional[GradScaler],
    scheduler: Optional[object],
    is_batch_scheduler: bool,
    epoch: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics_tracker = MetricsTracker(
        num_classes=len(train_loader.dataset.classes),
        class_names=train_loader.dataset.classes,
        score_metric=config.score_metric
    )
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    accumulation_steps = config.gradient_accumulation_steps
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if config.mixed_precision and scaler is not None:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if is_batch_scheduler and scheduler is not None:
                    scheduler.step()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                if is_batch_scheduler and scheduler is not None:
                    scheduler.step()
        
        # Get predictions
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
        
        # Update metrics
        metrics_tracker.update(preds, labels, probs, loss.item() * accumulation_steps)
        
        # Update progress bar
        if (batch_idx + 1) % config.log_every_n_steps == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "lr": f"{current_lr:.2e}"
            })
    
    return metrics_tracker.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    split_name: str = "val"
) -> Tuple[Dict[str, float], MetricsTracker]:
    """Evaluate model on a dataset."""
    model.eval()
    
    metrics_tracker = MetricsTracker(
        num_classes=len(data_loader.dataset.classes),
        class_names=data_loader.dataset.classes,
        score_metric=config.score_metric
    )
    
    pbar = tqdm(data_loader, desc=f"[{split_name.capitalize()}]", leave=False)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        if config.mixed_precision:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        # Update metrics
        metrics_tracker.update(preds, labels, probs, loss.item())
    
    return metrics_tracker.compute(), metrics_tracker


def save_training_results(
    output_dir: Path,
    metrics: Dict[str, float],
    metrics_tracker: MetricsTracker,
    class_names: List[str],
    split_name: str,
    epoch: Optional[int] = None
) -> None:
    """Save comprehensive training results."""
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    suffix = f"_epoch_{epoch:03d}" if epoch is not None else "_final"
    save_metrics(metrics, str(results_dir / f"{split_name}_metrics{suffix}.json"), epoch)
    
    # Save confusion matrix
    if "confusion_matrix" in metrics:
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            str(results_dir / f"{split_name}_confusion_matrix{suffix}.json")
        )
    
    # Save ROC curves if available
    if hasattr(metrics_tracker, 'all_probs') and len(metrics_tracker.all_probs) > 0:
        roc_curves = metrics_tracker.compute_roc_curves()
        if roc_curves:
            with open(results_dir / f"{split_name}_roc_curves{suffix}.json", 'w') as f:
                json.dump(roc_curves, f, indent=2)
    
    # Save classification report
    if hasattr(metrics_tracker, 'all_preds') and hasattr(metrics_tracker, 'all_labels'):
        save_classification_report(
            np.array(metrics_tracker.all_labels),
            np.array(metrics_tracker.all_preds),
            class_names,
            str(results_dir / f"{split_name}_classification_report{suffix}.txt")
        )


def train(config: Config, logger: logging.Logger) -> None:
    """Main training function."""
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(config, str(output_dir / "config.yaml"))
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(config)
    num_classes = len(class_names)
    logger.info(f"Found {num_classes} classes: {class_names}")
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Save class names
    with open(output_dir / "class_names.json", 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Compute class weights if needed
    class_weights = None
    if config.class_weights == "auto" or config.class_weights == "balanced":
        logger.info("Computing class weights...")
        class_weights = compute_class_weights(train_loader.dataset, num_classes, device)
        logger.info(f"Class weights: {class_weights.tolist()}")
    elif isinstance(config.class_weights, list):
        class_weights = torch.tensor(config.class_weights, device=device)
    
    # Create model
    logger.info(f"Creating model: {config.model}")
    model = create_model(config, num_classes, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = create_loss_function(config, class_weights)
    
    # Create optimizer
    optimizer = create_optimizer(config, model)
    
    # Create scheduler
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    scheduler, is_batch_scheduler = create_scheduler(config, optimizer, steps_per_epoch)
    
    # Setup mixed precision
    scaler = GradScaler('cuda') if config.mixed_precision and device.type == "cuda" else None
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if config.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        start_epoch, _ = load_checkpoint(
            config.resume_from_checkpoint, model, optimizer, scheduler, device
        )
        start_epoch += 1
    
    # Setup early stopping
    early_stopping = None
    if config.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            mode=config.early_stopping.mode
        )
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=str(output_dir / "checkpoints"),
        keep_top_k=config.keep_top_k_checkpoints,
        mode="max" if config.score_metric != "loss" else "min"
    )
    
    # Training history
    history = {
        "train_loss": [],
        "train_score": [],
        "val_loss": [],
        "val_score": [],
        "learning_rate": []
    }
    
    best_val_score = float("-inf")
    
    # Training loop
    logger.info("Starting training...")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config,
            scaler, scheduler, is_batch_scheduler, epoch, logger
        )
        
        # Validate
        val_metrics, val_tracker = evaluate(
            model, val_loader, criterion, device, config, "val"
        )
        
        # Get scores
        train_score = MetricsTracker(
            num_classes, class_names, config.score_metric
        ).get_score(train_metrics)
        val_score = val_tracker.get_score(val_metrics)
        
        # Update scheduler (epoch-based)
        if scheduler is not None and not is_batch_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train {config.score_metric}: {train_score:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val {config.score_metric}: {val_score:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_score"].append(train_score)
        history["val_loss"].append(val_metrics["loss"])
        history["val_score"].append(val_score)
        history["learning_rate"].append(current_lr)
        
        # Save checkpoint
        if epoch % config.save_every_n_epochs == 0:
            checkpoint_manager.save(
                model, optimizer, scheduler, epoch, val_score, config, class_names
            )
        
        # Check for best model
        if val_score > best_val_score:
            best_val_score = val_score
            logger.info(f"New best validation {config.score_metric}: {val_score:.4f}")
            
            # Save best model
            best_model_path = output_dir / "best_model.pth"
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state,
                "score": val_score,
                "class_names": class_names
            }, best_model_path)
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_score):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time / 60:.1f} minutes")
    
    # Save training history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model for final evaluation
    logger.info("Loading best model for final evaluation...")
    best_checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(best_checkpoint["model_state_dict"])
    else:
        model.load_state_dict(best_checkpoint["model_state_dict"])
    
    # Final validation evaluation
    logger.info("Running final validation evaluation...")
    val_metrics, val_tracker = evaluate(model, val_loader, criterion, device, config, "val")
    save_training_results(output_dir, val_metrics, val_tracker, class_names, "val")
    
    # Test evaluation
    logger.info("Running test evaluation...")
    test_metrics, test_tracker = evaluate(model, test_loader, criterion, device, config, "test")
    save_training_results(output_dir, test_metrics, test_tracker, class_names, "test")
    
    # Generate figures
    logger.info("Generating figures...")
    
    # Get ROC curves for test set
    test_roc_curves = None
    if hasattr(test_tracker, 'all_probs') and len(test_tracker.all_probs) > 0:
        test_roc_curves = test_tracker.compute_roc_curves()
    
    generate_all_figures(
        output_dir=output_dir,
        history=history,
        metrics=test_metrics,
        class_names=class_names,
        split_name="test",
        score_metric=config.score_metric,
        roc_curves=test_roc_curves
    )
    
    # Also generate validation figures
    val_roc_curves = None
    if hasattr(val_tracker, 'all_probs') and len(val_tracker.all_probs) > 0:
        val_roc_curves = val_tracker.compute_roc_curves()
    
    generate_all_figures(
        output_dir=output_dir,
        history=None,  # Only include training curves once
        metrics=val_metrics,
        class_names=class_names,
        split_name="val",
        score_metric=config.score_metric,
        roc_curves=val_roc_curves
    )
    
    logger.info(f"Figures saved to: {output_dir / 'figures'}")
    
    # Generate stratification distribution histograms (image-bin mode only)
    if config.image_bin_path and config.stratification_csv:
        logger.info("Generating stratification distribution histograms...")
        # _TransformSubset exposes .indices from the split
        plot_stratification_histograms(
            csv_path=config.stratification_csv,
            image_bin_path=config.image_bin_path,
            train_indices=train_loader.dataset.indices,
            val_indices=val_loader.dataset.indices,
            test_indices=test_loader.dataset.indices,
            samples=train_loader.dataset.dataset.samples,
            save_dir=output_dir / "figures" / "stratification_bins",
            stratify_columns=config.stratify_columns or None,
            stratify_bins=config.stratify_bins or None,
        )
    
    # Generate network visualizations
    logger.info("Generating network weight visualizations...")
    input_shape = (1, config.num_channels, config.image_size[0], config.image_size[1])
    generate_all_network_figures(
        model=model,
        save_dir=output_dir / "figures" / "network",
        input_shape=input_shape
    )
    
    # Log final results
    logger.info("=" * 60)
    logger.info("Final Results:")
    logger.info("=" * 60)
    logger.info(f"Validation {config.score_metric}: {val_tracker.get_score(val_metrics):.4f}")
    logger.info(f"Test {config.score_metric}: {test_tracker.get_score(test_metrics):.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test MCC: {test_metrics['mcc']:.4f}")
    
    # Export models
    if config.export_onnx:
        logger.info("Exporting model to ONNX...")
        onnx_path = str(output_dir / "model.onnx")
        input_shape = (1, config.num_channels, config.image_size[0], config.image_size[1])
        export_to_onnx(model, onnx_path, input_shape, config.onnx_opset_version, torch.device("cpu"))
        logger.info(f"ONNX model saved to: {onnx_path}")
    
    if config.export_torchscript:
        logger.info("Exporting model to TorchScript...")
        ts_path = str(output_dir / "model.pt")
        input_shape = (1, config.num_channels, config.image_size[0], config.image_size[1])
        export_to_torchscript(model, ts_path, input_shape, torch.device("cpu"))
        logger.info(f"TorchScript model saved to: {ts_path}")
    
    # Save final model
    if config.save_final_model:
        final_model_path = output_dir / "final_model.pth"
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({
            "model_state_dict": model_state,
            "class_names": class_names,
            "config": config.__dict__
        }, final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
    
    logger.info("Training complete!")


def main(argv: List[str]) -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a multi-class image classifier."
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (e.g., --override num_epochs=20 batch_size=32)"
    )
    
    args = parser.parse_args(argv)
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply overrides
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            if hasattr(config, key):
                # Try to parse as int, float, or bool
                field_type = type(getattr(config, key))
                if field_type == bool:
                    value = value.lower() in ("true", "1", "yes")
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                setattr(config, key, value)
    
    # Setup output directory
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config.log_level, output_dir)
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Output directory: {config.output_directory}")
    
    # Run training
    train(config, logger)


if __name__ == "__main__":
    main(sys.argv[1:])

