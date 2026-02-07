"""Model creation and utilities."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple

from .config import Config


# Mapping of model names to torchvision constructors and their classifier attribute names
MODEL_REGISTRY = {
    # ResNet family
    "resnet18": (models.resnet18, "fc", 512),
    "resnet34": (models.resnet34, "fc", 512),
    "resnet50": (models.resnet50, "fc", 2048),
    "resnet101": (models.resnet101, "fc", 2048),
    "resnet152": (models.resnet152, "fc", 2048),
    
    # VGG family
    "vgg11": (models.vgg11, "classifier", 4096),
    "vgg13": (models.vgg13, "classifier", 4096),
    "vgg16": (models.vgg16, "classifier", 4096),
    "vgg19": (models.vgg19, "classifier", 4096),
    
    # DenseNet family
    "densenet121": (models.densenet121, "classifier", 1024),
    "densenet169": (models.densenet169, "classifier", 1664),
    "densenet201": (models.densenet201, "classifier", 1920),
    
    # EfficientNet family
    "efficientnet_b0": (models.efficientnet_b0, "classifier", 1280),
    "efficientnet_b1": (models.efficientnet_b1, "classifier", 1280),
    "efficientnet_b2": (models.efficientnet_b2, "classifier", 1408),
    "efficientnet_b3": (models.efficientnet_b3, "classifier", 1536),
    "efficientnet_b4": (models.efficientnet_b4, "classifier", 1792),
    
    # MobileNet family
    "mobilenet_v2": (models.mobilenet_v2, "classifier", 1280),
    "mobilenet_v3_small": (models.mobilenet_v3_small, "classifier", 576),
    "mobilenet_v3_large": (models.mobilenet_v3_large, "classifier", 960),
    
    # Other models
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, "fc", 1024),
    "squeezenet1_0": (models.squeezenet1_0, "classifier", 512),
    "alexnet": (models.alexnet, "classifier", 4096),
    "googlenet": (models.googlenet, "fc", 1024),
    "inception_v3": (models.inception_v3, "fc", 2048),
}


def create_model(
    config: Config,
    num_classes: int,
    device: torch.device
) -> nn.Module:
    """Create a model based on configuration."""
    model_name = config.model.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    
    constructor, classifier_name, in_features = MODEL_REGISTRY[model_name]
    
    # Get pretrained weights
    if config.pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None
    
    # Create model
    model = constructor(weights=weights)
    
    # Modify input layer for grayscale images if needed
    if config.num_channels == 1:
        model = _modify_input_channels(model, model_name)
    
    # Modify classifier for our number of classes
    model = _modify_classifier(model, model_name, classifier_name, in_features, num_classes, config.dropout_rate)
    
    # Freeze backbone if requested
    if config.freeze_backbone:
        _freeze_backbone(model, classifier_name)
    elif config.freeze_layers > 0:
        _freeze_n_layers(model, config.freeze_layers)
    
    model = model.to(device)
    
    # Use DataParallel if requested
    if config.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model


def _modify_input_channels(model: nn.Module, model_name: str) -> nn.Module:
    """Modify the first convolutional layer to accept grayscale input."""
    if model_name.startswith("resnet") or model_name.startswith("shufflenet"):
        # ResNet and ShuffleNet have conv1 as first layer
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        # Initialize with mean of original weights
        if old_conv.weight is not None:
            model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    
    elif model_name.startswith("vgg") or model_name.startswith("alexnet"):
        old_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if old_conv.weight is not None:
            model.features[0].weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    
    elif model_name.startswith("densenet"):
        old_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if old_conv.weight is not None:
            model.features.conv0.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    
    elif model_name.startswith("efficientnet") or model_name.startswith("mobilenet"):
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if old_conv.weight is not None:
            model.features[0][0].weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    
    elif model_name.startswith("squeezenet"):
        old_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if old_conv.weight is not None:
            model.features[0].weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    
    elif model_name in ("googlenet", "inception_v3"):
        old_conv = model.conv1.conv
        model.conv1.conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if old_conv.weight is not None:
            model.conv1.conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    
    return model


def _modify_classifier(
    model: nn.Module,
    model_name: str,
    classifier_name: str,
    in_features: int,
    num_classes: int,
    dropout_rate: float
) -> nn.Module:
    """Modify the classifier head for the target number of classes."""
    if classifier_name == "fc":
        # Simple fully connected layer (ResNet, GoogLeNet, etc.)
        old_fc = getattr(model, classifier_name)
        if hasattr(old_fc, 'in_features'):
            in_features = old_fc.in_features
        
        setattr(model, classifier_name, nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        ))
    
    elif classifier_name == "classifier":
        if model_name.startswith("vgg") or model_name == "alexnet":
            # VGG/AlexNet have a sequential classifier
            old_classifier = model.classifier
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7 if model_name == "alexnet" else 512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(4096, num_classes)
            )
        elif model_name.startswith("efficientnet") or model_name.startswith("mobilenet"):
            old_classifier = model.classifier
            in_features = old_classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        elif model_name.startswith("densenet"):
            old_classifier = model.classifier
            in_features = old_classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        elif model_name.startswith("squeezenet"):
            # SqueezeNet uses Conv2d as final classifier
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
    
    return model


def _freeze_backbone(model: nn.Module, classifier_name: str) -> None:
    """Freeze all parameters except the classifier."""
    for name, param in model.named_parameters():
        if classifier_name not in name:
            param.requires_grad = False


def _freeze_n_layers(model: nn.Module, n_layers: int) -> None:
    """Freeze the first n layers of the model."""
    layer_count = 0
    for child in model.children():
        if layer_count < n_layers:
            for param in child.parameters():
                param.requires_grad = False
            layer_count += 1
        else:
            break


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, int, int, int],
    opset_version: int = 18,
    device: torch.device = torch.device("cpu")
) -> None:
    """Export model to ONNX format."""
    model.eval()
    model = model.to(device)
    
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Use dynamo=False to avoid dynamic_shapes requirement
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )


def export_to_torchscript(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, int, int, int],
    device: torch.device = torch.device("cpu")
) -> None:
    """Export model to TorchScript format."""
    model.eval()
    model = model.to(device)
    
    dummy_input = torch.randn(*input_shape, device=device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(save_path)
