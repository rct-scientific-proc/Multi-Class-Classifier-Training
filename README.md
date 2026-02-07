# MultiClassifierTraining

A comprehensive PyTorch-based framework for training multi-class image classifiers with YAML configuration, extensive metrics, and visualization capabilities.

## Features

- **Configurable via YAML** - All training parameters, model selection, augmentation, and export options in one file
- **Multiple Architectures** - Support for ResNet, VGG, DenseNet, EfficientNet, MobileNet, and more
- **Rich Data Augmentation** - Rotation, affine transforms, color jitter, random erasing, Mixup, CutMix
- **Advanced Training** - Mixed precision (AMP), gradient accumulation, learning rate schedulers, early stopping
- **Multiple Loss Functions** - Cross-entropy, label smoothing, focal loss, weighted cross-entropy
- **Comprehensive Metrics** - Accuracy, F1, precision, recall, MCC, balanced accuracy, ROC-AUC, confusion matrix
- **Visualization** - Training curves, confusion matrices, ROC curves, per-class accuracy, network weight visualization
- **Model Export** - ONNX and TorchScript export for deployment

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional but recommended)

### Quick Setup

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Windows (Command Prompt):**
```batch
setup.bat
```

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

#### Custom Python Path and CUDA Version

```powershell
# PowerShell - specify Python path and CUDA version
.\setup.ps1 -PythonPath "C:\Python314\python.exe" -CudaVersion "cu126"

# Command Prompt - specify CUDA version (auto-detects Python)
setup.bat cu126

# Command Prompt - specify both Python path and CUDA version
setup.bat "C:\Python314\python.exe" cu126
```

Supported CUDA versions: `cu126`, `cu128`, `cu130` (default: `cu126`)

### Manual Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Download and partition MNIST dataset:

```bash
python scripts/download_mnist.py
```

This creates the following structure:
```
data/mnist/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ...
├── validate/
│   └── ...
└── test/
    └── ...
```

### 2. Configure Training

Edit `example_config.yaml` or create your own configuration file. Key settings:

```yaml
model: "resnet18"
pretrained: true
train_directory: "data/mnist/train"
val_directory: "data/mnist/validate"
test_directory: "data/mnist/test"
output_directory: "results/mnist_resnet18"
batch_size: 64
num_epochs: 10
learning_rate: 0.001
```

### 3. Train

```bash
python train.py --config example_config.yaml
```

## Configuration Reference

### Model Options

```yaml
# Supported models
model: "resnet18"  # resnet18, resnet34, resnet50, resnet101, resnet152
                   # vgg16, vgg19, vgg16_bn, vgg19_bn
                   # densenet121, densenet169, densenet201
                   # efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
                   # mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
                   # shufflenet_v2_x1_0, squeezenet1_0
                   # inception_v3, googlenet, alexnet

pretrained: true           # Use ImageNet pretrained weights
freeze_backbone: false     # Freeze all backbone layers
freeze_layers: 0           # Freeze first N layers
dropout_rate: 0.5          # Classifier head dropout
```

### Data Configuration

```yaml
train_directory: "data/train"
val_directory: "data/validate"
test_directory: "data/test"
output_directory: "results/experiment1"

image_size: [224, 224]     # Resize images to this size
num_channels: 3            # 1 for grayscale, 3 for RGB
num_workers: 4             # DataLoader workers
pin_memory: true           # Faster GPU transfer

normalize_mean: [0.485, 0.456, 0.406]  # ImageNet normalization
normalize_std: [0.229, 0.224, 0.225]
```

### Data Augmentation

```yaml
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.0
  rotation: 15             # Random rotation ±15 degrees
  
  affine:
    enabled: true
    translate: [0.1, 0.1]
    scale: [0.9, 1.1]
    shear: 10
  
  color_jitter:
    enabled: true
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  
  random_erasing:
    enabled: false
    probability: 0.5
  
  mixup:
    enabled: false
    alpha: 0.2
  
  cutmix:
    enabled: false
    alpha: 1.0
```

### Training Parameters

```yaml
batch_size: 64
eval_batch_size: 128
num_epochs: 100
gradient_accumulation_steps: 1
max_grad_norm: 1.0         # Gradient clipping
mixed_precision: true      # FP16 training
seed: 42                   # Reproducibility
```

### Optimizer

```yaml
optimizer: "AdamW"         # Adam, AdamW, SGD, RMSprop, Adagrad
learning_rate: 0.001
weight_decay: 0.0001

sgd:
  momentum: 0.9
  nesterov: true

adam:
  betas: [0.9, 0.999]
  eps: 1.0e-8
  amsgrad: false
```

### Learning Rate Schedulers

```yaml
learning_rate_scheduler: "CosineAnnealingLR"
# Options: StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
#          CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR

warmup_epochs: 5
warmup_start_factor: 0.1

cosine_annealing_lr:
  T_max: 100
  eta_min: 1.0e-6

reduce_lr_on_plateau:
  mode: "max"
  factor: 0.1
  patience: 5
```

### Loss Functions

```yaml
loss_function: "CrossEntropyLoss"
# Options: CrossEntropyLoss, LabelSmoothingCrossEntropy, FocalLoss, WeightedCrossEntropyLoss

label_smoothing: 0.1

focal_loss:
  alpha: 1.0
  gamma: 2.0

class_weights: "auto"      # "auto", "balanced", or [w1, w2, ...]
```

### Metrics & Evaluation

```yaml
score_metric: "f1_macro"   # Primary metric for model selection
# Options: accuracy, f1_score, f1_macro, f1_weighted, precision, recall,
#          mcc, auc_roc, balanced_accuracy

additional_metrics:
  - "accuracy"
  - "f1_macro"
  - "precision"
  - "recall"
  - "confusion_matrix"
  - "per_class_accuracy"

compute_roc_curves: true
```

### Early Stopping & Checkpointing

```yaml
save_every_n_epochs: 1
keep_top_k_checkpoints: 3
save_final_model: true
resume_from_checkpoint: null

early_stopping:
  enabled: true
  patience: 10
  min_delta: 0.001
  mode: "max"
```

### Model Export

```yaml
export_onnx: true
onnx_opset_version: 18
export_torchscript: false

quantization:
  enabled: false
  backend: "fbgemm"        # "fbgemm" for x86, "qnnpack" for ARM
```

### Hardware

```yaml
device: "cuda"             # cuda, cpu, mps, or cuda:0
data_parallel: false       # Multi-GPU DataParallel

distributed:
  enabled: false
  backend: "nccl"
```

## Output Structure

After training, the output directory contains:

```
results/experiment_name/
├── checkpoints/
│   ├── best_model.pth
│   ├── epoch_010.pth
│   └── final_model.pth
├── figures/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── roc_curves.png
│   ├── per_class_accuracy.png
│   ├── metrics_summary.png
│   └── network/
│       ├── conv_filter_weights.png
│       ├── weight_distributions.png
│       ├── layer_statistics.png
│       └── network_architecture.png
├── model.onnx
├── metrics.json
├── confusion_matrix.json
├── classification_report.json
├── config.yaml
└── train.log
```

## Metrics

The framework computes and logs:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions / total |
| Balanced Accuracy | Average of per-class recall |
| F1 Macro | Unweighted mean of per-class F1 |
| F1 Weighted | Weighted mean of per-class F1 |
| Precision | True positives / predicted positives |
| Recall | True positives / actual positives |
| MCC | Matthews Correlation Coefficient |
| AUC-ROC | Area under ROC curve (per class) |
| Confusion Matrix | Per-class prediction breakdown |

## Visualizations

### Training Curves

Training and validation loss/accuracy over epochs with best epoch marked.

### Confusion Matrix
Actual vs predicted class heatmap, available in raw counts and normalized formats.

### ROC Curves
Per-class ROC curves with AUC values for multi-class classification.

### Network Visualization
- **Convolutional Filters**: First-layer filters visualized as image grids
- **Weight Distributions**: Histograms of layer weights with mean/std
- **Layer Statistics**: Bar charts of parameter counts and weight statistics
- **Architecture Diagram**: Visual representation of network layers

## Project Structure

```
MultiClassifierTraining/
├── train.py                 # Main training script
├── example_config.yaml      # Example configuration
├── requirements.txt         # Python dependencies
├── setup.ps1               # Windows PowerShell setup
├── setup.bat               # Windows Command Prompt setup
├── setup.sh                # Linux/macOS setup
├── scripts/
│   └── download_mnist.py   # MNIST dataset downloader
├── utils/
│   ├── config.py           # Configuration dataclasses
│   ├── data.py             # Dataset and DataLoader utilities
│   ├── metrics.py          # Metrics tracking and computation
│   ├── models.py           # Model creation and export
│   ├── network_viz.py      # Network weight visualization
│   ├── plotting.py         # Training visualization plots
│   └── training.py         # Training loop utilities
├── data/                   # Dataset directory
├── results/                # Training outputs
└── output/                 # Legacy output directory
```

## Tips

### For Small Datasets
- Use `pretrained: true` with ImageNet weights
- Set `freeze_backbone: true` or `freeze_layers: 10+`
- Enable strong augmentation (rotation, affine, color jitter)
- Use `class_weights: "auto"` for imbalanced classes

### For Large Datasets
- Disable freezing to train full network
- Use larger models (ResNet50, EfficientNet-B2)
- Enable mixed precision for faster training
- Increase `num_workers` and `batch_size`

### For Grayscale Images
- Set `num_channels: 1`
- Images are automatically converted to 3-channel for pretrained models


## License

MIT License

