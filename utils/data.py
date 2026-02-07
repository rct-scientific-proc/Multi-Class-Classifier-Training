"""Data loading and augmentation utilities."""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config, AugmentationConfig


class ImageFolderDataset(Dataset):
    """Custom ImageFolder dataset that handles grayscale conversion."""
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        num_channels: int = 3
    ):
        self.root = Path(root)
        self.transform = transform
        self.num_channels = num_channels
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        self._load_samples()
    
    def _load_samples(self) -> None:
        """Load all image paths and their labels."""
        # Get sorted class directories
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path)
        
        # Convert to RGB or grayscale as needed
        if self.num_channels == 3:
            image = image.convert('RGB')
        elif self.num_channels == 1:
            image = image.convert('L')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def build_transforms(
    config: Config,
    is_training: bool = True
) -> transforms.Compose:
    """Build transforms based on configuration."""
    transform_list = []
    aug = config.augmentation
    
    # Resize
    transform_list.append(transforms.Resize(config.image_size))
    
    if is_training:
        # Random horizontal flip
        if aug.horizontal_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=aug.horizontal_flip))
        
        # Random vertical flip
        if aug.vertical_flip > 0:
            transform_list.append(transforms.RandomVerticalFlip(p=aug.vertical_flip))
        
        # Random rotation
        if aug.rotation > 0:
            transform_list.append(transforms.RandomRotation(degrees=aug.rotation))
        
        # Random affine
        if aug.affine.get('enabled', False):
            transform_list.append(transforms.RandomAffine(
                degrees=0,
                translate=tuple(aug.affine.get('translate', [0.1, 0.1])),
                scale=tuple(aug.affine.get('scale', [0.9, 1.1])),
                shear=aug.affine.get('shear', 10)
            ))
        
        # Color jitter (only for RGB)
        if config.num_channels == 3 and aug.color_jitter.get('enabled', False):
            transform_list.append(transforms.ColorJitter(
                brightness=aug.color_jitter.get('brightness', 0.2),
                contrast=aug.color_jitter.get('contrast', 0.2),
                saturation=aug.color_jitter.get('saturation', 0.2),
                hue=aug.color_jitter.get('hue', 0.1)
            ))
    
    # To tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize - adjust mean/std for grayscale
    if config.num_channels == 1:
        # Use single channel normalization
        mean = [sum(config.normalize_mean) / 3]
        std = [sum(config.normalize_std) / 3]
    else:
        mean = config.normalize_mean
        std = config.normalize_std
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    # Random erasing (after normalization)
    if is_training and aug.random_erasing.get('enabled', False):
        transform_list.append(transforms.RandomErasing(
            p=aug.random_erasing.get('probability', 0.5),
            scale=tuple(aug.random_erasing.get('scale', [0.02, 0.33])),
            ratio=tuple(aug.random_erasing.get('ratio', [0.3, 3.3]))
        ))
    
    return transforms.Compose(transform_list)


def create_data_loaders(
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test data loaders."""
    
    # Build transforms
    train_transform = build_transforms(config, is_training=True)
    eval_transform = build_transforms(config, is_training=False)
    
    # Create datasets
    train_dataset = ImageFolderDataset(
        config.train_directory,
        transform=train_transform,
        num_channels=config.num_channels
    )
    val_dataset = ImageFolderDataset(
        config.val_directory,
        transform=eval_transform,
        num_channels=config.num_channels
    )
    test_dataset = ImageFolderDataset(
        config.test_directory,
        transform=eval_transform,
        num_channels=config.num_channels
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def compute_class_weights(
    dataset: Dataset,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    class_counts = torch.zeros(num_classes)
    
    # Count labels directly from dataset samples (no image loading)
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    # Compute inverse frequency weights
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return weights.to(device)
