# -*- coding: utf-8 -*-
"""
Standalone GradCAM utility for classic CNN models (ResNet18, ResNet50, DenseNet, EfficientNet)
Takes a specific WSI directory containing tiles as input
Consistent with existing codebase functionality
"""

import torch
import torch.nn.functional as F
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAM:
    """GradCAM implementation for classic CNN models"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks on the target layer"""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found in model")
        
        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate GradCAM for a single image"""
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Calculate weights
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Generate CAM
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                           size=input_image.shape[2:], 
                           mode='bilinear', 
                           align_corners=False)
        cam = cam.squeeze().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

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

def calculate_grid_dimensions_from_coordinates(tile_coords, tile_size=224):
    """
    Calculate grid dimensions from tile coordinates
    Assumes coordinates are absolute pixel positions, not grid indices
    """
    if not tile_coords:
        return (1, 1)  # Default fallback
    
    # Find the grid dimensions
    x_coords = [coord[0] for coord in tile_coords]
    y_coords = [coord[1] for coord in tile_coords]
    
    # Calculate grid size based on coordinate ranges
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Calculate grid dimensions (coordinates are absolute positions)
    # Add tile_size to account for the last tile
    grid_width = (max_x // tile_size) + 1
    grid_height = (max_y // tile_size) + 1
    
    print(f"Grid dimensions: {grid_width} x {grid_height} (based on coordinates {min_x}-{max_x}, {min_y}-{max_y})")
    
    return (grid_width, grid_height)

def create_wsi_heatmap_from_tiles_grid(tile_cams, tile_coords, tile_size=224, output_size=(1000, 1000)):
    """
    Create WSI-level heatmap from individual tile GradCAMs using a grid approach
    Fills missing tiles with blank CAMs to ensure complete coverage
    """
    # Calculate grid dimensions
    grid_width, grid_height = calculate_grid_dimensions_from_coordinates(tile_coords, tile_size)
    
    # Create a dictionary to map coordinates to CAMs
    coord_to_cam = {}
    for cam, (x, y) in zip(tile_cams, tile_coords):
        coord_to_cam[(x, y)] = cam
    
    print(f"Found {len(coord_to_cam)} unique tile coordinates")
    
    # Create empty heatmap with grid dimensions
    heatmap_width = grid_width * tile_size
    heatmap_height = grid_height * tile_size
    heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    
    print(f"Creating heatmap of size {heatmap_width} x {heatmap_height}")
    
    # Fill the heatmap grid by grid
    filled_tiles = 0
    missing_tiles = 0
    
    for grid_y in range(grid_height):
        for grid_x in range(grid_width):
            # Calculate tile coordinates in the grid (absolute pixel positions)
            tile_x = grid_x * tile_size
            tile_y = grid_y * tile_size
            
            # Check if we have a CAM for this position
            if (tile_x, tile_y) in coord_to_cam:
                # Use the actual CAM
                cam = coord_to_cam[(tile_x, tile_y)]
                cam_resized = cv2.resize(cam, (tile_size, tile_size))
                heatmap[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size] = cam_resized
                filled_tiles += 1
            else:
                # Fill with blank CAM (zeros)
                heatmap[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size] = 0
                missing_tiles += 1
    
    print(f"Filled {filled_tiles} tiles with CAMs, {missing_tiles} tiles with blanks")
    
    # Normalize heatmap (only consider non-zero regions)
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Resize to output size
    heatmap = cv2.resize(heatmap, output_size)
    
    return heatmap

def create_wsi_heatmap_from_tiles(tile_cams, tile_coords, tile_size=224, output_size=(1000, 1000)):
    """
    Create WSI-level heatmap from individual tile GradCAMs
    Automatically determines slide size from coordinates
    """
    # Calculate slide size from coordinates
    slide_size = calculate_slide_size_from_coordinates(tile_coords, tile_size)
    print(f"Calculated slide size: {slide_size}")
    
    # Create empty heatmap
    heatmap = np.zeros(slide_size, dtype=np.float32)
    
    # Place GradCAM scores at tile locations
    valid_tiles = 0
    for i, (cam, (x, y)) in enumerate(zip(tile_cams, tile_coords)):
        # Convert coordinates to heatmap indices
        x_idx = int(x)
        y_idx = int(y)
        
        # Debug: print some coordinate info
        if i < 5 or i % 500 == 0:
            print(f"Tile {i}: coordinates ({x_idx}, {y_idx}), slide_size {slide_size}")
        
        # Ensure coordinates are within bounds
        if 0 <= x_idx < slide_size[0] and 0 <= y_idx < slide_size[1]:
            # Resize CAM to tile size
            cam_resized = cv2.resize(cam, (tile_size, tile_size))
            
            # Calculate end indices
            x_end = min(x_idx + tile_size, slide_size[0])
            y_end = min(y_idx + tile_size, slide_size[1])
            
            # Calculate CAM end indices
            cam_x_end = min(tile_size, x_end - x_idx)
            cam_y_end = min(tile_size, y_end - y_idx)
            
            # Debug: check for empty slices
            if x_end <= x_idx or y_end <= y_idx:
                print(f"Warning: Empty slice detected for tile {i} at ({x_idx}, {y_idx})")
                continue
                
            if cam_x_end <= 0 or cam_y_end <= 0:
                print(f"Warning: Invalid CAM dimensions for tile {i}: cam_x_end={cam_x_end}, cam_y_end={cam_y_end}")
                continue
            
            # Ensure we have valid slices
            if x_end > x_idx and y_end > y_idx and cam_x_end > 0 and cam_y_end > 0:
                # Add CAM to the tile region
                try:
                    heatmap[y_idx:y_end, x_idx:x_end] += cam_resized[:cam_y_end, :cam_x_end]
                    valid_tiles += 1
                except ValueError as e:
                    print(f"Error adding tile {i} at ({x_idx}, {y_idx}): {e}")
                    print(f"  heatmap slice shape: {heatmap[y_idx:y_end, x_idx:x_end].shape}")
                    print(f"  cam slice shape: {cam_resized[:cam_y_end, :cam_x_end].shape}")
                    continue
        else:
            print(f"Warning: Tile {i} coordinates ({x_idx}, {y_idx}) out of bounds for slide size {slide_size}")
    
    print(f"Successfully processed {valid_tiles} out of {len(tile_cams)} tiles")
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Resize to output size
    heatmap = cv2.resize(heatmap, output_size)
    
    return heatmap

def calculate_slide_size_from_coordinates(tile_coords, tile_size=224, padding=100):
    """
    Calculate slide size from tile coordinates
    Consistent with how the system handles spatial layout
    """
    if not tile_coords:
        return (1000, 1000)  # Default fallback
    
    max_x = max(coord[0] for coord in tile_coords)
    max_y = max(coord[1] for coord in tile_coords)
    
    # Add tile size and padding to get full slide dimensions
    slide_width = max_x + tile_size + padding
    slide_height = max_y + tile_size + padding
    
    return (slide_width, slide_height)

def generate_gradcam_for_wsi_directory(wsi_tile_dir, model_path, model_name, num_classes, 
                                     output_dir=None, tile_size=None, target_class=None, use_grid=True):
    """
    Generate GradCAM heatmap for a specific WSI directory containing tiles
    
    Args:
        wsi_tile_dir: Directory containing tile images for a specific WSI
        model_path: Path to the trained model file
        model_name: Name of the model architecture (resnet18, resnet50, densenet, efficient)
        num_classes: Number of classes in the model
        output_dir: Directory to save results (if None, creates in wsi_tile_dir)
        tile_size: Size of each tile (if None, will be determined from model)
        target_class: Specific class to generate GradCAM for (if None, uses predicted class)
        use_grid: Whether to use grid-based approach (fills missing tiles with blanks)
    
    Returns:
        dict: Dictionary containing paths to generated files
    """
    if not os.path.exists(wsi_tile_dir):
        raise ValueError(f"WSI tile directory does not exist: {wsi_tile_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(wsi_tile_dir, 'gradcam_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model, input_size = load_model(model_path, model_name, num_classes)
    
    if tile_size is None:
        tile_size = input_size
    
    # Get target layer name
    target_layer_name = get_target_layer_name(model_name)
    print(f"Using target layer: {target_layer_name}")
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer_name)
    
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
    
    # Generate GradCAM for each tile
    tile_cams = []
    tile_coords = []
    tile_predictions = []
    
    print("Generating GradCAM for tiles...")
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
            
            tile_predictions.append({
                'path': tile_path,
                'prediction': prediction,
                'probabilities': probabilities
            })
            
            # Generate GradCAM
            if target_class is not None:
                cam_target = target_class
            else:
                cam_target = prediction
            
            cam = grad_cam.generate_cam(input_tensor, cam_target)
            tile_cams.append(cam)
            
            # Extract coordinates
            x, y = extract_tile_coordinates_from_path(tile_path)
            tile_coords.append((x, y))
            
        except Exception as e:
            print(f"Error processing tile {tile_path}: {e}")
            continue
    
    if not tile_cams:
        raise ValueError("No valid tiles processed")
    
    # Create WSI heatmap
    print("Creating WSI-level heatmap...")
    if use_grid:
        wsi_heatmap = create_wsi_heatmap_from_tiles_grid(
            tile_cams, 
            tile_coords, 
            tile_size=tile_size
        )
    else:
        wsi_heatmap = create_wsi_heatmap_from_tiles(
            tile_cams, 
            tile_coords, 
            tile_size=tile_size
        )
    
    # Save results
    wsi_name = os.path.basename(wsi_tile_dir)
    
    # Save WSI heatmap
    heatmap_path = os.path.join(output_dir, f"{wsi_name}_gradcam_heatmap.png")
    plt.figure(figsize=(12, 10))
    plt.imshow(wsi_heatmap, cmap='jet')
    plt.colorbar(label='GradCAM Score')
    plt.title(f'GradCAM Heatmap - {wsi_name}')
    plt.axis('off')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw heatmap data
    heatmap_npy_path = os.path.join(output_dir, f"{wsi_name}_gradcam_heatmap.npy")
    np.save(heatmap_npy_path, wsi_heatmap)
    
    # Save tile-level predictions
    predictions_df = pd.DataFrame(tile_predictions)
    predictions_path = os.path.join(output_dir, f"{wsi_name}_tile_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save individual tile GradCAMs
    tile_cam_dir = os.path.join(output_dir, f"{wsi_name}_tile_cams")
    os.makedirs(tile_cam_dir, exist_ok=True)
    
    print("Saving individual tile GradCAMs...")
    for i, (cam, tile_path) in enumerate(zip(tile_cams, [t['path'] for t in tile_predictions])):
        tile_name = os.path.basename(tile_path).replace('.jpg', '_gradcam.png')
        tile_cam_path = os.path.join(tile_cam_dir, tile_name)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(cam, cmap='jet')
        plt.colorbar(label='GradCAM Score')
        plt.title(f'Tile GradCAM - {os.path.basename(tile_path)}')
        plt.axis('off')
        plt.savefig(tile_cam_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Clean up
    grad_cam.remove_hooks()
    
    # Return paths to generated files
    results = {
        'wsi_heatmap': heatmap_path,
        'wsi_heatmap_npy': heatmap_npy_path,
        'tile_predictions': predictions_path,
        'tile_cams_dir': tile_cam_dir,
        'output_dir': output_dir
    }
    
    print(f"GradCAM analysis complete. Results saved to: {output_dir}")
    return results

def main():
    """Command line interface for standalone GradCAM generation"""
    parser = argparse.ArgumentParser(description='Generate GradCAM heatmaps for WSI tiles')
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
                       help='Output directory (default: wsi_dir/gradcam_results)')
    parser.add_argument('--tile_size', type=int, default=None,
                       help='Tile size (default: model input size)')
    parser.add_argument('--target_class', type=int, default=None,
                       help='Target class for GradCAM (default: predicted class)')
    parser.add_argument('--use_grid', action='store_true', default=True,
                       help='Use grid-based approach (fills missing tiles with blanks)')
    
    args = parser.parse_args()
    
    # Generate GradCAM
    results = generate_gradcam_for_wsi_directory(
        wsi_tile_dir=args.wsi_dir,
        model_path=args.model_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        target_class=args.target_class,
        use_grid=args.use_grid
    )
    
    print("\nGenerated files:")
    for key, path in results.items():
        print(f"{key}: {path}")

if __name__ == "__main__":
    main()