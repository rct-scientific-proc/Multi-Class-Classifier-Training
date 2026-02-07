"""Configuration loading and validation utilities."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotation: float = 15.0
    affine: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "translate": [0.1, 0.1],
        "scale": [0.9, 1.1],
        "shear": 10
    })
    color_jitter: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    })
    random_erasing: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "probability": 0.5,
        "scale": [0.02, 0.33],
        "ratio": [0.3, 3.3]
    })
    mixup: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "alpha": 0.2})
    cutmix: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "alpha": 1.0})


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001
    mode: str = "max"


@dataclass
class Config:
    """Training configuration."""
    # Model
    model: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    freeze_layers: int = 0
    dropout_rate: float = 0.5
    
    # Data paths
    train_directory: str = "data/mnist/train"
    val_directory: str = "data/mnist/validate"
    test_directory: str = "data/mnist/test"
    output_directory: str = "results"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    num_channels: int = 3
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Training
    batch_size: int = 64
    eval_batch_size: int = 128
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    seed: int = 42
    
    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    sgd: Dict[str, Any] = field(default_factory=lambda: {"momentum": 0.9, "nesterov": True})
    adam: Dict[str, Any] = field(default_factory=lambda: {"betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False})
    
    # Scheduler
    learning_rate_scheduler: str = "CosineAnnealingLR"
    warmup_epochs: int = 0
    warmup_start_factor: float = 0.1
    step_lr: Dict[str, Any] = field(default_factory=lambda: {"step_size": 5, "gamma": 0.1})
    multi_step_lr: Dict[str, Any] = field(default_factory=lambda: {"milestones": [30, 60, 90], "gamma": 0.1})
    exponential_lr: Dict[str, Any] = field(default_factory=lambda: {"gamma": 0.95})
    cosine_annealing_lr: Dict[str, Any] = field(default_factory=lambda: {"T_max": 10, "eta_min": 1e-6})
    cosine_annealing_warm_restarts: Dict[str, Any] = field(default_factory=lambda: {"T_0": 10, "T_mult": 2, "eta_min": 1e-6})
    reduce_lr_on_plateau: Dict[str, Any] = field(default_factory=lambda: {"mode": "max", "factor": 0.1, "patience": 5, "threshold": 0.0001, "min_lr": 1e-7})
    one_cycle_lr: Dict[str, Any] = field(default_factory=lambda: {"max_lr": 0.01, "pct_start": 0.3, "anneal_strategy": "cos", "div_factor": 25.0, "final_div_factor": 10000.0})
    
    # Loss
    loss_function: str = "CrossEntropyLoss"
    label_smoothing: float = 0.1
    focal_loss: Dict[str, Any] = field(default_factory=lambda: {"alpha": 1.0, "gamma": 2.0})
    class_weights: Union[str, List[float]] = "auto"
    
    # Metrics
    score_metric: str = "accuracy"
    additional_metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro"])
    compute_roc_curves: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 1
    keep_top_k_checkpoints: int = 3
    save_final_model: bool = True
    resume_from_checkpoint: Optional[str] = None
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    
    # Logging
    log_level: str = "INFO"
    log_every_n_steps: int = 10
    tensorboard: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "log_dir": "runs"})
    wandb: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "project": "multiclass-classifier", "entity": None, "run_name": None, "tags": []})
    save_predictions: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "num_samples": 16, "save_every_n_epochs": 5})
    
    # Export
    export_onnx: bool = True
    onnx_opset_version: int = 18
    export_torchscript: bool = False
    quantization: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "backend": "fbgemm"})
    
    # Hardware
    device: str = "cuda"
    data_parallel: bool = False
    distributed: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "backend": "nccl", "world_size": 1, "rank": 0})


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Handle nested configs
    if 'augmentation' in yaml_config:
        aug_dict = yaml_config.pop('augmentation')
        yaml_config['augmentation'] = AugmentationConfig(**{
            k: v for k, v in aug_dict.items() 
            if k in AugmentationConfig.__dataclass_fields__
        })
    
    if 'early_stopping' in yaml_config:
        es_dict = yaml_config.pop('early_stopping')
        yaml_config['early_stopping'] = EarlyStoppingConfig(**es_dict)
    
    # Filter out unknown keys
    valid_keys = Config.__dataclass_fields__.keys()
    filtered_config = {k: v for k, v in yaml_config.items() if k in valid_keys}
    
    return Config(**filtered_config)


def save_config(config: Config, save_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = {}
    for key, value in config.__dict__.items():
        if hasattr(value, '__dict__'):
            config_dict[key] = value.__dict__
        else:
            config_dict[key] = value
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
