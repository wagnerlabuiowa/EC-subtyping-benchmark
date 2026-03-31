# -*- coding: utf-8 -*-
"""
Standalone subtype prediction heatmap utility for classic CNN models (ResNet18, ResNet50, DenseNet, EfficientNet)
Takes a specific WSI directory containing tiles as input
Colors tiles based on predicted subtype
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import os
import re
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse
import glob
from matplotlib.patches import Patch
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_target_layer_name(model_name):
    """Get the appropriate target layer name for different model architectures"""
    if 'resnet18' in model_name.lower():
        return 'layer4'
    elif 'resnet50' in model_name.lower():
        return 'layer4'
    elif 'densenet' in model_name.lower():
        return 'features.denseblock4'
    elif 'efficient' in model_name.lower():
        return '_blocks.16'  # Last block before global pooling
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def extract_tile_coordinates_from_path(tile_path):
    """
    Extract tile coordinates from tile file path
    Handle multiple possible formats for tile naming
    """
    filename = os.path.basename(tile_path)
    
    # Try different coordinate extraction patterns
    patterns = [
        r'\((\d+),(\d+)\)',  # (x,y) format
        r'_(\d+)_(\d+)',     # _x_y format
        r'(\d+)_(\d+)',      # x_y format
        r'x(\d+)_y(\d+)',    # x123_y456 format
    ]
    
    for pattern in patterns:
        coord_match = re.search(pattern, filename)
        if coord_match:
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))
            return x, y
    
    # If no coordinates found, try to extract from path structure
    # This handles cases where coordinates might be in directory names
    path_parts = tile_path.split(os.sep)
    for part in path_parts:
        for pattern in patterns:
            coord_match = re.search(pattern, part)
            if coord_match:
                x = int(coord_match.group(1))
                y = int(coord_match.group(2))
                return x, y
    
    # Fallback: use file index as coordinates
    print(f"Warning: No coordinates found in {tile_path}, using fallback coordinates")
    return 0, 0

def load_model(model_path, model_name, num_classes):
    """Load a trained model using the same method as the codebase"""
    import utils.utils as utils
    
    model, input_size = utils.Initialize_model(
        model_name=model_name, 
        num_classes=num_classes, 
        feature_extract=False, 
        use_pretrained=False
    )
    
    # Load model weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    return model, input_size

def create_subtype_heatmap_from_tiles(
    tile_predictions, tile_coords, tile_size=224, num_classes=2, cmap_name="Pastel1"
):
    """
    Returns:
        rgb_map: np.ndarray, shape (H, W, 3), dtype uint8
        class_map: np.ndarray, shape (H, W), dtype int
    """
    if not tile_coords:
        print("No coordinates found, creating default heatmap")
        return np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=int)

    # Get sorted unique x and y coordinates
    x_coords_sorted = sorted(set([coord[0] for coord in tile_coords]))
    y_coords_sorted = sorted(set([coord[1] for coord in tile_coords]))
    x_index = {x: i for i, x in enumerate(x_coords_sorted)}
    y_index = {y: i for i, y in enumerate(y_coords_sorted)}
    grid_width = len(x_coords_sorted)
    grid_height = len(y_coords_sorted)

    print(f"Grid: {grid_width} x {grid_height}")

    # Create a 2D array of class indices, -1 for background
    class_map = np.full((grid_height, grid_width), fill_value=-1, dtype=int)

    for pred, (x, y) in zip(tile_predictions, tile_coords):
        gx = x_index[x]
        gy = y_index[y]
        class_map[gy, gx] = pred

    # Convert class_map to RGB using the colormap, with white for background
    cmap = plt.get_cmap(cmap_name, num_classes)
    rgb_map = np.ones((grid_height, grid_width, 3), dtype=np.float32)  # white background
    mask = class_map >= 0
    rgb_map[mask] = cmap(class_map[mask])[:, :3]  # Only color where class is valid

    # Scale to 0-255 and convert to uint8
    rgb_map = (rgb_map * 255).astype(np.uint8)
    return rgb_map, class_map

def save_top_k_tiles_per_class(tile_results, k, output_dir, class_names=None):
    """
    Save the top-k tiles for each class based on predicted probability.
    tile_results: list of dicts with keys 'path', 'prediction', 'probabilities'
    k: number of top tiles to save per class
    output_dir: directory to save the top tiles
    class_names: list of class names (optional)
    """
    df = pd.DataFrame(tile_results)
    num_classes = len(df['probabilities'][0])
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    top_tiles_root = os.path.join(output_dir, "top_tiles")
    os.makedirs(top_tiles_root, exist_ok=True)

    for class_idx in range(num_classes):
        class_dir = os.path.join(top_tiles_root, class_names[class_idx])
        os.makedirs(class_dir, exist_ok=True)
        # Get probability for this class
        df['class_prob'] = df['probabilities'].apply(lambda x: x[class_idx])
        # Sort and take top k
        topk = df.sort_values('class_prob', ascending=False).head(k)
        for i, row in topk.iterrows():
            tile_path = row['path']
            prob = row['class_prob']
            fname = os.path.basename(tile_path)
            out_fname = f"{prob:.3f}_{fname}"
            shutil.copy(tile_path, os.path.join(class_dir, out_fname))

def generate_subtype_heatmap_for_wsi_directory(wsi_tile_dir, model_path, model_name, num_classes, 
                                             output_dir=None, tile_size=None, class_names=None, output_size=(1000, 1000)):
    """
    Generate subtype prediction heatmap for a specific WSI directory containing tiles
    
    Args:
        wsi_tile_dir: Directory containing tile images for a specific WSI
        model_path: Path to the trained model file
        model_name: Name of the model architecture (resnet18, resnet50, densenet, efficient)
        num_classes: Number of classes in the model
        output_dir: Directory to save results (if None, creates in wsi_tile_dir)
        tile_size: Size of each tile (if None, will be determined from model)
        class_names: List of class names for legend (optional)
    
    Returns:
        dict: Dictionary containing paths to generated files
    """
    if not os.path.exists(wsi_tile_dir):
        raise ValueError(f"WSI tile directory does not exist: {wsi_tile_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(wsi_tile_dir, 'subtype_heatmap_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model, input_size = load_model(model_path, model_name, num_classes)
    
    if tile_size is None:
        tile_size = input_size
    
    # Get all tile files
    tile_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    tile_files = []
    for ext in tile_extensions:
        tile_files.extend(glob.glob(os.path.join(wsi_tile_dir, ext)))
    
    if not tile_files:
        raise ValueError(f"No tile files found in directory: {wsi_tile_dir}")
    
    print(f"Found {len(tile_files)} tiles")
    
    # Prepare transform - consistent with DatasetLoader_Classic
    transform = transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor()
    ])
    
    # Generate predictions for each tile
    tile_predictions = []
    tile_coords = []
    tile_results = []
    
    print("Generating predictions for tiles...")
    for tile_path in tqdm(tile_files):
        try:
            # Load and preprocess tile - consistent with DatasetLoader_Classic
            image = Image.open(tile_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.argmax(dim=1).item()
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            tile_results.append({
                'path': tile_path,
                'prediction': prediction,
                'probabilities': probabilities
            })
            
            tile_predictions.append(prediction)
            
            # Extract coordinates
            x, y = extract_tile_coordinates_from_path(tile_path)
            tile_coords.append((x, y))
            
        except Exception as e:
            print(f"Error processing tile {tile_path}: {e}")
            continue
    
    if not tile_predictions:
        raise ValueError("No valid tiles processed")
    
    # Create WSI heatmap
    print("Creating WSI-level subtype heatmap...")
    rgb_map, class_map = create_subtype_heatmap_from_tiles(
        tile_predictions, 
        tile_coords, 
        tile_size=tile_size,
        num_classes=num_classes
    )

    # Upscale to output size
    img = Image.fromarray(rgb_map)
    img = img.resize(output_size, resample=Image.NEAREST)
    wsi_heatmap = np.array(img)

    # Save results
    wsi_name = os.path.basename(wsi_tile_dir)
    heatmap_path = os.path.join(output_dir, f"{wsi_name}_subtype_heatmap.png")

    # Use the standard Pastel1 colormap
    cmap = plt.get_cmap("Pastel1")

    plt.figure(figsize=(15, 12))
    ax = plt.gca()

    # Plot a white background
    ax.imshow(np.ones(class_map.shape + (3,), dtype=np.float32), vmin=0, vmax=1)

    # Mask for valid tiles
    mask = class_map >= 0
    # Create an RGBA image from the colormap
    rgba_map = cmap(class_map.clip(0, 8))  # Only indices 0-8 are valid for Pastel1
    rgba_map[..., -1] = mask.astype(float)  # Alpha channel: 1 for tiles, 0 for background

    # Overlay the class map
    ax.imshow(rgba_map, vmin=0, vmax=1)

    plt.title(f'Subtype Prediction Heatmap - {wsi_name}', fontsize=16)
    plt.axis('off')

    legend_elements = [
        Patch(facecolor=cmap(i), label=class_names[i]) for i in range(num_classes)
    ]
    #legend_elements.append(Patch(facecolor='white', label='Background/No Tissue'))
    plt.legend(
        handles=legend_elements, 
        loc='upper left', 
        bbox_to_anchor=(1.05, 1.0), 
        borderaxespad=0.0,
        frameon=False
    )
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw heatmap data
    heatmap_npy_path = os.path.join(output_dir, f"{wsi_name}_subtype_heatmap.npy")
    np.save(heatmap_npy_path, wsi_heatmap)
    
    # Save tile-level predictions
    predictions_df = pd.DataFrame(tile_results)
    predictions_path = os.path.join(output_dir, f"{wsi_name}_tile_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    # Create summary statistics
    unique_predictions, counts = np.unique(tile_predictions, return_counts=True)
    summary_stats = []
    for pred, count in zip(unique_predictions, counts):
        percentage = (count / len(tile_predictions)) * 100
        class_name = class_names[pred] if class_names else f'Class {pred}'
        summary_stats.append({
            'class': pred,
            'class_name': class_name,
            'count': count,
            'percentage': percentage
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, f"{wsi_name}_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save top-k tiles for each class
    save_top_k_tiles_per_class(
        tile_results=tile_results,
        k=8,  # or any value you want
        output_dir=output_dir,
        class_names=class_names
    )

    # Print summary
    print(f"\nSubtype distribution for {wsi_name}:")
    for stat in summary_stats:
        print(f"  {stat['class_name']}: {stat['count']} tiles ({stat['percentage']:.1f}%)")
    
    # Return paths to generated files
    results = {
        'wsi_heatmap': heatmap_path,
        'wsi_heatmap_npy': heatmap_npy_path,
        'tile_predictions': predictions_path,
        'summary_stats': summary_path,
        'output_dir': output_dir
    }
    
    print(f"Subtype heatmap analysis complete. Results saved to: {output_dir}")
    return results

def main():
    """Command line interface for standalone subtype heatmap generation"""
    parser = argparse.ArgumentParser(description='Generate subtype prediction heatmaps for WSI tiles')
    parser.add_argument('--wsi_dir', type=str, required=True,
                       help='Directory containing WSI tiles')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['resnet18', 'resnet50', 'densenet', 'efficient'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='Number of classes in the model')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: wsi_dir/subtype_heatmap_results)')
    parser.add_argument('--tile_size', type=int, default=None,
                       help='Tile size (default: model input size)')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Names of the classes (e.g., "Subtype1" "Subtype2")')
    
    args = parser.parse_args()
    
    # Generate subtype heatmap
    results = generate_subtype_heatmap_for_wsi_directory(
        wsi_tile_dir=args.wsi_dir,
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        class_names=args.class_names
    )
    
    print("\nGenerated files:")
    for key, path in results.items():
        print(f"{key}: {path}")

if __name__ == "__main__":
    main()