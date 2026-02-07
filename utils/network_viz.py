"""Neural network weight and architecture visualization utilities."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def visualize_conv_weights(
    model: nn.Module,
    save_path: str,
    max_filters: int = 64,
    normalize: bool = True
) -> None:
    """
    Visualize convolutional layer weights as image grids.
    
    Args:
        model: The neural network model
        save_path: Path to save the figure
        max_filters: Maximum number of filters to display per layer
        normalize: Whether to normalize weights to [0, 1] range
    """
    # Find all conv layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    if not conv_layers:
        print("No convolutional layers found in model")
        return
    
    # Create figure with subplots for each conv layer (up to 4)
    num_layers = min(len(conv_layers), 4)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
    
    if num_layers == 1:
        axes = [axes]
    
    for idx, (name, conv) in enumerate(conv_layers[:num_layers]):
        weights = conv.weight.data.cpu().numpy()
        
        # Get dimensions
        out_channels, in_channels, kh, kw = weights.shape
        
        # Limit number of filters
        num_filters = min(out_channels, max_filters)
        
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_filters)))
        
        # For first conv layer with 3 input channels, show as RGB
        if in_channels == 3 and idx == 0:
            # Create grid image
            grid_img = np.zeros((grid_size * kh, grid_size * kw, 3))
            
            for i in range(num_filters):
                row = i // grid_size
                col = i % grid_size
                
                # Get filter and normalize
                filt = weights[i].transpose(1, 2, 0)  # HWC format
                
                if normalize:
                    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                
                grid_img[row*kh:(row+1)*kh, col*kw:(col+1)*kw] = filt
            
            axes[idx].imshow(grid_img)
            
        elif in_channels == 1:
            # Grayscale input - show filters directly
            grid_img = np.zeros((grid_size * kh, grid_size * kw))
            
            for i in range(num_filters):
                row = i // grid_size
                col = i % grid_size
                
                filt = weights[i, 0]
                
                if normalize:
                    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                
                grid_img[row*kh:(row+1)*kh, col*kw:(col+1)*kw] = filt
            
            axes[idx].imshow(grid_img, cmap='viridis')
            
        else:
            # Multiple input channels - average across channels
            grid_img = np.zeros((grid_size * kh, grid_size * kw))
            
            for i in range(num_filters):
                row = i // grid_size
                col = i % grid_size
                
                filt = weights[i].mean(axis=0)  # Average across input channels
                
                if normalize:
                    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                
                grid_img[row*kh:(row+1)*kh, col*kw:(col+1)*kw] = filt
            
            axes[idx].imshow(grid_img, cmap='viridis')
        
        # Truncate long names
        short_name = name if len(name) < 20 else "..." + name[-17:]
        axes[idx].set_title(f'{short_name}\n{out_channels} filters, {kh}x{kw}', fontsize=10)
        axes[idx].axis('off')
    
    plt.suptitle('Convolutional Filter Weights', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_weight_distributions(
    model: nn.Module,
    save_path: str,
    max_layers: int = 12
) -> None:
    """
    Visualize weight distributions as histograms for each layer.
    
    Args:
        model: The neural network model
        save_path: Path to save the figure
        max_layers: Maximum number of layers to display
    """
    # Collect weights from all layers
    layer_weights = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            weights = param.data.cpu().numpy().flatten()
            # Get a shorter name
            short_name = name.replace('.weight', '').replace('module.', '')
            if len(short_name) > 25:
                short_name = "..." + short_name[-22:]
            # Prefix purely numeric names to avoid matplotlib warning
            try:
                float(short_name)
                short_name = f"layer_{short_name}"
            except ValueError:
                pass
            layer_weights.append((short_name, weights))
    
    # Limit number of layers
    layer_weights = layer_weights[:max_layers]
    num_layers = len(layer_weights)
    
    if num_layers == 0:
        print("No weight layers found")
        return
    
    # Create figure
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (name, weights) in enumerate(layer_weights):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        # Plot histogram
        ax.hist(weights, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean = np.mean(weights)
        std = np.std(weights)
        ax.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'μ={mean:.3f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1)
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1)
        
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Weight Value', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
    
    # Hide empty subplots
    for idx in range(num_layers, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.suptitle('Weight Distributions by Layer', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_layer_statistics(
    model: nn.Module,
    save_path: str
) -> None:
    """
    Create a bar chart showing weight statistics per layer.
    
    Args:
        model: The neural network model
        save_path: Path to save the figure
    """
    # Collect statistics
    layer_stats = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            weights = param.data.cpu().numpy()
            
            short_name = name.replace('.weight', '').replace('module.', '')
            if len(short_name) > 20:
                short_name = "..." + short_name[-17:]
            # Prefix purely numeric names to avoid matplotlib warning
            try:
                float(short_name)
                short_name = f"layer_{short_name}"
            except ValueError:
                pass
            
            layer_stats.append({
                'name': short_name,
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'params': weights.size
            })
    
    if not layer_stats:
        print("No weight layers found")
        return
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Prefix purely numeric names to avoid matplotlib parsing them as numbers
    names = []
    for s in layer_stats:
        name = s['name']
        # Check if name could be parsed as a number
        try:
            float(name)
            name = f"layer_{name}"  # Prefix numeric-only names
        except ValueError:
            pass
        names.append(name)
    x = np.arange(len(names))
    
    # Plot 1: Mean and Std
    ax1 = axes[0]
    means = [s['mean'] for s in layer_stats]
    stds = [s['std'] for s in layer_stats]
    
    ax1.bar(x - 0.2, means, 0.4, label='Mean', color='steelblue', alpha=0.8)
    ax1.bar(x + 0.2, stds, 0.4, label='Std Dev', color='coral', alpha=0.8)
    ax1.set_ylabel('Value')
    ax1.set_title('Weight Mean and Standard Deviation per Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Parameter counts
    ax2 = axes[1]
    params = [s['params'] for s in layer_stats]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(params)))
    
    bars = ax2.bar(x, params, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('Parameter Count per Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, p in zip(bars, params):
        height = bar.get_height()
        if height > max(params) * 0.1:
            ax2.annotate(f'{p:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_network_architecture(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    max_layers: int = 20
) -> None:
    """
    Create a simplified network architecture diagram.
    
    Args:
        model: The neural network model
        save_path: Path to save the figure
        input_shape: Input tensor shape (batch, channels, height, width)
        max_layers: Maximum layers to show
    """
    # Collect layer information
    layers_info = []
    
    def get_layer_type(module):
        name = module.__class__.__name__
        if 'Conv2d' in name:
            return 'Conv', 'lightblue'
        elif 'Linear' in name:
            return 'FC', 'lightgreen'
        elif 'BatchNorm' in name:
            return 'BN', 'lightyellow'
        elif 'ReLU' in name or 'GELU' in name:
            return 'Act', 'lightcoral'
        elif 'Pool' in name or 'pool' in name.lower():
            return 'Pool', 'plum'
        elif 'Dropout' in name:
            return 'Drop', 'lightgray'
        else:
            return name[:8], 'white'
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_type, color = get_layer_type(module)
            
            # Get shape info if available
            shape_info = ""
            if hasattr(module, 'weight') and module.weight is not None:
                shape = list(module.weight.shape)
                shape_info = f"{shape}"
            
            layers_info.append({
                'name': name,
                'type': layer_type,
                'color': color,
                'shape': shape_info,
                'module': module.__class__.__name__
            })
    
    # Limit and sample layers if too many
    if len(layers_info) > max_layers:
        step = len(layers_info) // max_layers
        layers_info = layers_info[::step][:max_layers]
    
    # Create figure
    num_layers = len(layers_info)
    fig_height = max(6, num_layers * 0.4)
    fig, ax = plt.subplots(1, 1, figsize=(12, fig_height))
    
    # Draw layers as boxes
    box_height = 0.6
    y_spacing = 1.0
    box_width = 3.0
    
    for idx, layer in enumerate(layers_info):
        y = (num_layers - idx - 1) * y_spacing
        
        # Draw box
        rect = FancyBboxPatch(
            (0, y), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add layer type text
        ax.text(box_width / 2, y + box_height / 2, layer['type'],
               ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Add layer name and info
        short_name = layer['name'] if len(layer['name']) < 30 else "..." + layer['name'][-27:]
        ax.text(box_width + 0.3, y + box_height / 2 + 0.1, short_name,
               ha='left', va='center', fontsize=8, fontfamily='monospace')
        ax.text(box_width + 0.3, y + box_height / 2 - 0.15, layer['shape'],
               ha='left', va='center', fontsize=7, color='gray')
        
        # Draw arrow to next layer
        if idx < num_layers - 1:
            arrow = FancyArrowPatch(
                (box_width / 2, y),
                (box_width / 2, y - y_spacing + box_height),
                arrowstyle='-|>',
                mutation_scale=15,
                color='gray',
                linewidth=1.5
            )
            ax.add_patch(arrow)
    
    # Add input/output labels
    ax.text(box_width / 2, num_layers * y_spacing + 0.3, 
           f'Input: {list(input_shape)}', ha='center', fontsize=10, fontweight='bold')
    ax.text(box_width / 2, -0.8, 'Output: [batch, num_classes]', 
           ha='center', fontsize=10, fontweight='bold')
    
    # Add legend
    legend_items = [
        ('Conv', 'lightblue'),
        ('FC', 'lightgreen'),
        ('BN', 'lightyellow'),
        ('Act', 'lightcoral'),
        ('Pool', 'plum'),
    ]
    for i, (label, color) in enumerate(legend_items):
        rect = Rectangle((8 + (i % 3) * 1.5, num_layers * y_spacing - 0.5 - (i // 3) * 0.6), 
                         0.4, 0.4, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(8.5 + (i % 3) * 1.5, num_layers * y_spacing - 0.3 - (i // 3) * 0.6, 
               label, fontsize=9, va='center')
    
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-1.5, num_layers * y_spacing + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.suptitle(f'Network Architecture ({model.__class__.__name__})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_network_figures(
    model: nn.Module,
    save_dir: Path,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
) -> None:
    """
    Generate all network visualization figures.
    
    Args:
        model: The trained model
        save_dir: Directory to save figures
        input_shape: Model input shape
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating network visualizations...")
    
    # Convolutional filter weights
    visualize_conv_weights(
        model,
        str(save_dir / "conv_filter_weights.png")
    )
    
    # Weight distributions
    visualize_weight_distributions(
        model,
        str(save_dir / "weight_distributions.png")
    )
    
    # Layer statistics
    visualize_layer_statistics(
        model,
        str(save_dir / "layer_statistics.png")
    )
    
    # Network architecture
    visualize_network_architecture(
        model,
        str(save_dir / "network_architecture.png"),
        input_shape=input_shape
    )
    
    print(f"Network visualizations saved to: {save_dir}")
