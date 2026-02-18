"""Data loading and augmentation utilities."""

import csv
import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config, AugmentationConfig

logger = logging.getLogger(__name__)


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


class _TransformSubset(Dataset):
    """A subset of a dataset with its own transform applied."""
    
    def __init__(
        self,
        dataset: ImageFolderDataset,
        indices: List[int],
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        # Expose .samples and .classes for downstream compatibility
        self.samples = [dataset.samples[i] for i in indices]
        self.classes = dataset.classes
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.dataset.samples[self.indices[idx]]
        
        image = Image.open(img_path)
        if self.dataset.num_channels == 3:
            image = image.convert('RGB')
        elif self.dataset.num_channels == 1:
            image = image.convert('L')
        
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


def _split_indices_stratified(
    samples: List[Tuple[Path, int]],
    stratify_groups: Dict[int, str],
    train_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split sample indices into train/val/test with stratification.
    
    Args:
        samples: List of (path, class_idx) tuples.
        stratify_groups: Mapping from sample index -> stratify group string.
            Indices not present or mapped to "" are treated as ungrouped.
        train_ratio: Fraction of data for training.
        test_ratio: Fraction of data for testing.
        seed: Random seed.
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.RandomState(seed)
    val_ratio = 1.0 - train_ratio - test_ratio

    # Organise indices by (class_label, stratify_group)
    grouped: Dict[Tuple[int, str], List[int]] = {}
    ungrouped: List[int] = []

    for idx, (_, class_idx) in enumerate(samples):
        group = stratify_groups.get(idx, "")
        if group == "":
            ungrouped.append(idx)
        else:
            key = (class_idx, group)
            grouped.setdefault(key, []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    # Split each stratification group proportionally
    for key, indices in grouped.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(round(n * train_ratio)))
        n_test = max(0, int(round(n * test_ratio)))
        n_val = n - n_train - n_test
        if n_val < 0:
            n_test = n - n_train
            n_val = 0
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Split ungrouped randomly
    if ungrouped:
        rng.shuffle(ungrouped)
        n = len(ungrouped)
        n_train = int(round(n * train_ratio))
        n_test = int(round(n * test_ratio))
        train_indices.extend(ungrouped[:n_train])
        val_indices.extend(ungrouped[n_train:n_train + (n - n_train - n_test)])
        test_indices.extend(ungrouped[n_train + (n - n_train - n_test):])

    return train_indices, val_indices, test_indices


def _split_indices_random(
    num_samples: int,
    train_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split indices randomly into train/val/test."""
    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    n_train = int(round(num_samples * train_ratio))
    n_test = int(round(num_samples * test_ratio))
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:num_samples - n_test].tolist()
    test_indices = indices[num_samples - n_test:].tolist()
    return train_indices, val_indices, test_indices


def _value_to_bin(value: float, bin_edges: List[float]) -> str:
    """Map a continuous value to a bin label given a list of edges."""
    for i in range(len(bin_edges) - 1):
        if value <= bin_edges[i + 1] or i == len(bin_edges) - 2:
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            return f"{lo:.4g}_to_{hi:.4g}"
    lo = bin_edges[-2]
    hi = bin_edges[-1]
    return f"{lo:.4g}_to_{hi:.4g}"


def _load_stratification_csv(
    csv_path: str,
    image_bin_path: str,
    samples: List[Tuple[Path, int]],
    stratify_columns: Optional[List[str]] = None,
    stratify_bins: Optional[Dict[str, int]] = None,
) -> Dict[int, str]:
    """Load a stratification CSV and map sample indices to stratify groups.
    
    The CSV must have a "filename" column whose values are paths relative to
    image_bin_path (or absolute paths, or just bare filenames).
    
    Additional columns are the stratification dimensions.  Which columns to
    use is controlled by *stratify_columns*:
      - If a non-empty list is given, only those column names are used.
      - If empty or None, **all** columns except "filename" are used.
    
    Columns listed in *stratify_bins* (mapping column name -> number of bins)
    are treated as continuous numeric values.  The loader reads all values
    for each such column, computes equal-width bins from min to max, and
    converts each value to a bin label automatically.  Columns **not** in
    stratify_bins are treated as categorical (values used as-is).
    
    When multiple stratification columns are present the group key is the
    pipe-separated concatenation of their values
    (e.g. ``"angle=bin2|quality=high"``).
    """
    bin_path = Path(image_bin_path).resolve()
    
    # Build a lookup: resolved_path -> sample_index
    path_to_idx: Dict[Path, int] = {}
    name_to_idx: Dict[str, int] = {}
    for idx, (p, _) in enumerate(samples):
        path_to_idx[p.resolve()] = idx
        # Also store by relative-to-bin path (normalised)
        try:
            rel = p.resolve().relative_to(bin_path)
            path_to_idx[rel] = idx
            # Store string version with forward slashes for matching
            path_to_idx[Path(str(rel).replace("\\", "/"))] = idx
        except ValueError:
            pass
        name_to_idx[p.name] = idx

    stratify_groups: Dict[int, str] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames:
            raise ValueError(
                f"Stratification CSV must have a 'filename' column. "
                f"Found: {reader.fieldnames}"
            )
        
        # Determine which columns to stratify on
        available_cols = [c for c in reader.fieldnames if c != "filename"]
        if stratify_columns:
            missing = [c for c in stratify_columns if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"Stratification columns {missing} not found in CSV. "
                    f"Available non-filename columns: {available_cols}"
                )
            strat_cols = stratify_columns
        else:
            strat_cols = available_cols
        
        if not strat_cols:
            raise ValueError(
                "No stratification columns found. The CSV must have at least "
                "one column besides 'filename', or specify stratify_columns in config."
            )
        
        bins_cfg = stratify_bins or {}
        binned_cols = {col for col in strat_cols if col in bins_cfg}
        if binned_cols:
            logger.info(
                f"Stratifying on column(s): {strat_cols}  "
                f"(continuous -> binned: {{{', '.join(f'{c}: {bins_cfg[c]} bins' for c in sorted(binned_cols))}}})"
            )
        else:
            logger.info(f"Stratifying on column(s): {strat_cols} (all categorical)")
        
        # --- First pass: collect raw rows and numeric values for binning ---
        raw_rows: List[Tuple[str, Dict[str, str]]] = []  # (filename, {col: raw_value})
        col_values: Dict[str, List[float]] = {col: [] for col in binned_cols}
        
        for row in reader:
            fname = row["filename"].strip()
            vals = {col: row[col].strip() for col in strat_cols}
            raw_rows.append((fname, vals))
            for col in binned_cols:
                v = vals[col]
                if v:
                    try:
                        col_values[col].append(float(v))
                    except ValueError:
                        pass  # non-numeric -> will be treated as empty
        
        # --- Build bin edges for continuous columns ---
        col_bin_edges: Dict[str, List[float]] = {}
        for col in binned_cols:
            n_bins = bins_cfg[col]
            values = col_values[col]
            if not values:
                logger.warning(f"Column '{col}' has no numeric values; skipping binning.")
                continue
            vmin, vmax = min(values), max(values)
            if vmin == vmax:
                col_bin_edges[col] = [vmin, vmax + 1.0]
            else:
                col_bin_edges[col] = [
                    vmin + i * (vmax - vmin) / n_bins for i in range(n_bins + 1)
                ]
            logger.info(
                f"Column '{col}': range [{vmin}, {vmax}] -> {n_bins} bins"
            )
        
        # --- Second pass: assign group keys ---
        matched = 0
        for fname, vals in raw_rows:
            # Build composite group key from all stratify columns
            parts = []
            all_empty = True
            for col in strat_cols:
                raw_val = vals[col]
                if not raw_val:
                    parts.append(f"{col}=" if len(strat_cols) > 1 else "")
                    continue
                all_empty = False
                if col in col_bin_edges:
                    # Continuous column -> bin it
                    try:
                        fval = float(raw_val)
                        edges = col_bin_edges[col]
                        bin_label = _value_to_bin(fval, edges)
                        parts.append(f"{col}={bin_label}" if len(strat_cols) > 1 else bin_label)
                    except ValueError:
                        parts.append(f"{col}={raw_val}" if len(strat_cols) > 1 else raw_val)
                else:
                    # Categorical column -> use as-is
                    parts.append(f"{col}={raw_val}" if len(strat_cols) > 1 else raw_val)
            group = "" if all_empty else "|".join(parts)
            
            # Try multiple matching strategies
            idx = None
            # 1) Exact resolved path
            resolved = Path(fname).resolve() if not Path(fname).is_absolute() else Path(fname)
            if resolved in path_to_idx:
                idx = path_to_idx[resolved]
            # 2) Relative to bin
            if idx is None:
                rel = Path(fname.replace("\\", "/"))
                if rel in path_to_idx:
                    idx = path_to_idx[rel]
            # 3) Try joining with bin path
            if idx is None:
                joined = (bin_path / fname).resolve()
                if joined in path_to_idx:
                    idx = path_to_idx[joined]
            # 4) Bare filename match (ambiguous but useful fallback)
            if idx is None:
                bare = Path(fname).name
                if bare in name_to_idx:
                    idx = name_to_idx[bare]
            
            if idx is not None:
                stratify_groups[idx] = group
                matched += 1

        logger.info(
            f"Stratification CSV: matched {matched}/{len(samples)} images to stratify groups, "
            f"{len(samples) - matched} will be randomly distributed."
        )

    return stratify_groups


def create_data_loaders(
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test data loaders."""
    
    # Build transforms
    train_transform = build_transforms(config, is_training=True)
    eval_transform = build_transforms(config, is_training=False)
    
    if config.image_bin_path is not None:
        # -- Image-bin mode: single directory split into train/val/test --
        logger.info(f"Using image bin: {config.image_bin_path}")
        
        # Load all images from the bin (with training transform as placeholder;
        # we'll override per-subset below via wrapper datasets)
        full_dataset = ImageFolderDataset(
            config.image_bin_path,
            transform=None,  # set per subset
            num_channels=config.num_channels,
        )
        
        if config.stratification_csv is not None:
            logger.info(f"Stratified splitting using: {config.stratification_csv}")
            stratify_groups = _load_stratification_csv(
                config.stratification_csv,
                config.image_bin_path,
                full_dataset.samples,
                stratify_columns=config.stratify_columns or None,
                stratify_bins=config.stratify_bins or None,
            )
            train_idx, val_idx, test_idx = _split_indices_stratified(
                full_dataset.samples,
                stratify_groups,
                config.train_val_split,
                config.test_split,
                seed=getattr(config, "seed", 42),
            )
        else:
            logger.info("Random splitting (no stratification CSV).")
            train_idx, val_idx, test_idx = _split_indices_random(
                len(full_dataset),
                config.train_val_split,
                config.test_split,
                seed=getattr(config, "seed", 42),
            )
        
        logger.info(
            f"Split sizes -- train: {len(train_idx)}, "
            f"val: {len(val_idx)}, test: {len(test_idx)}"
        )
        
        # Wrap subsets with the appropriate transforms
        train_dataset = _TransformSubset(full_dataset, train_idx, train_transform)
        val_dataset = _TransformSubset(full_dataset, val_idx, eval_transform)
        test_dataset = _TransformSubset(full_dataset, test_idx, eval_transform)
        
        class_names = full_dataset.classes
    else:
        # -- Standard mode: separate train/val/test directories --
        train_dataset = ImageFolderDataset(
            config.train_directory,
            transform=train_transform,
            num_channels=config.num_channels,
        )
        val_dataset = ImageFolderDataset(
            config.val_directory,
            transform=eval_transform,
            num_channels=config.num_channels,
        )
        test_dataset = ImageFolderDataset(
            config.test_directory,
            transform=eval_transform,
            num_channels=config.num_channels,
        )
        class_names = train_dataset.classes
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    return train_loader, val_loader, test_loader, class_names


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
