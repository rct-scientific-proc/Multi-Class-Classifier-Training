"""
Script to download MNIST digits dataset and save images as PNGs for testing.

Usage:
    python scripts/download_mnist.py --output ./data/mnist

This will create the following structure:
    ./data/mnist/
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
"""

import argparse
import os
from pathlib import Path

from torchvision import datasets
from PIL import Image


def download_and_save_mnist(output_dir: str, train: bool = True, max_per_class: int = None):
    """
    Download MNIST dataset and save images as PNGs.
    
    Args:
        output_dir: Output directory path
        train: If True, download training set; otherwise download test set
        max_per_class: Maximum number of images to save per class (None for all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create class directories (0-9)
    for digit in range(10):
        (output_path / str(digit)).mkdir(exist_ok=True)
    
    # Download MNIST dataset
    print(f"Downloading MNIST {'training' if train else 'test'} dataset...")
    dataset = datasets.MNIST(
        root="./data/raw",
        train=train,
        download=True
    )
    
    # Track counts per class
    class_counts = {i: 0 for i in range(10)}
    
    print(f"Saving images to {output_path}...")
    for idx, (image, label) in enumerate(dataset):
        # Check if we've reached max for this class
        if max_per_class is not None and class_counts[label] >= max_per_class:
            continue
        
        # Save image as PNG
        filename = f"{class_counts[label]:05d}.png"
        save_path = output_path / str(label) / filename
        image.save(save_path)
        
        class_counts[label] += 1
        
        # Progress update
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} images...")
        
        # Check if all classes have reached max
        if max_per_class is not None and all(c >= max_per_class for c in class_counts.values()):
            break
    
    # Print summary
    print("\nSummary:")
    print(f"  Output directory: {output_path}")
    total_saved = sum(class_counts.values())
    print(f"  Total images saved: {total_saved}")
    print("  Images per class:")
    for digit in range(10):
        print(f"    {digit}: {class_counts[digit]}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MNIST digits dataset and save as PNG images."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./data/mnist",
        help="Output directory for images. Default: ./data/mnist"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Download test set instead of training set"
    )
    
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of images to save per class (default: all)"
    )
    
    args = parser.parse_args()
    
    download_and_save_mnist(
        output_dir=args.output,
        train=not args.test,
        max_per_class=args.max_per_class
    )
    
    print("\nDone! You can now use this dataset with:")
    print(f"  python main.py --idp {args.output} --model resnet18 --output ./output/model.pth")


if __name__ == "__main__":
    main()
