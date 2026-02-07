"""
Script to download MNIST digits dataset and save images as PNGs for testing.

Usage:
    python scripts/download_mnist.py --output ./data/mnist

This will create the following structure:
    ./data/mnist/
        train/
            0/
                00000.png
                00001.png
                ...
            1/
                00000.png
                ...
            ...
            9/
                ...
        validate/
            0/
            1/
            ...
            9/
        test/
            0/
            1/
            ...
            9/
"""

import argparse
import random
from pathlib import Path

from torchvision import datasets


def download_and_save_mnist(
    output_dir: str,
    val_split: float = 0.2,
    max_per_class: int = None,
    seed: int = 42
):
    """
    Download MNIST dataset and save images as PNGs partitioned into train/validate/test.
    
    Args:
        output_dir: Output directory path
        val_split: Fraction of training data to use for validation (default: 0.1)
        max_per_class: Maximum number of images to save per class per split (None for all)
        seed: Random seed for reproducible train/val split
    """
    random.seed(seed)
    output_path = Path(output_dir)
    
    # Create directory structure: {split}/{digit}
    splits = ["train", "validate", "test"]
    for split in splits:
        for digit in range(10):
            (output_path / split / str(digit)).mkdir(parents=True, exist_ok=True)
    
    # Download MNIST datasets
    print("Downloading MNIST training dataset...")
    train_dataset = datasets.MNIST(root="./data/raw", train=True, download=True)
    
    print("Downloading MNIST test dataset...")
    test_dataset = datasets.MNIST(root="./data/raw", train=False, download=True)
    
    # Group training data by label for stratified split
    train_by_label = {i: [] for i in range(10)}
    for idx, (image, label) in enumerate(train_dataset):
        train_by_label[label].append((idx, image))
    
    # Split training data into train and validate (stratified)
    train_images = {i: [] for i in range(10)}
    val_images = {i: [] for i in range(10)}
    
    for label, images in train_by_label.items():
        random.shuffle(images)
        val_count = int(len(images) * val_split)
        val_images[label] = images[:val_count]
        train_images[label] = images[val_count:]
    
    # Group test data by label
    test_images = {i: [] for i in range(10)}
    for idx, (image, label) in enumerate(test_dataset):
        test_images[label].append((idx, image))
    
    # Save images for each split
    summary = {}
    for split_name, split_data in [("train", train_images), ("validate", val_images), ("test", test_images)]:
        print(f"\nSaving {split_name} images...")
        class_counts = {i: 0 for i in range(10)}
        
        for label in range(10):
            images = split_data[label]
            if max_per_class is not None:
                images = images[:max_per_class]
            
            for img_idx, (_, image) in enumerate(images):
                filename = f"{img_idx:05d}.png"
                save_path = output_path / split_name / str(label) / filename
                image.save(save_path)
                class_counts[label] += 1
        
        summary[split_name] = class_counts
        total = sum(class_counts.values())
        print(f"  Saved {total} images to {output_path / split_name}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"Output directory: {output_path}")
    print(f"Validation split: {val_split * 100:.1f}%")
    print()
    
    for split_name in splits:
        counts = summary[split_name]
        total = sum(counts.values())
        print(f"{split_name.capitalize():10} - Total: {total:6} images")
        print(f"            Per class: {dict(counts)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download MNIST digits dataset and save as PNG images partitioned into train/validate/test."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./data/mnist",
        help="Output directory for images. Default: ./data/mnist"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation. Default: 0.2"
    )
    
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of images to save per class per split (default: all)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split. Default: 42"
    )
    
    args = parser.parse_args()
    
    download_and_save_mnist(
        output_dir=args.output,
        val_split=args.val_split,
        max_per_class=args.max_per_class,
        seed=args.seed
    )
    
    print("\nDone! You can now use this dataset with:")
    print(f"  python main.py --idp {args.output}/train --model resnet18 --output ./output/model.pth")


if __name__ == "__main__":
    main()
