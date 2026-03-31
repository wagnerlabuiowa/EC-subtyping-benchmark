
from pathlib import Path
from glob import glob
import joblib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tiatoolbox.utils.misc import imread

# ── CONFIG — update these paths for your environment ───
TILE_DIR   = Path("path/to/toptiles/")                 # input folder with tile images
OUT_DIR    = Path("path/to/hovernet_output/")           # directory with .dat prediction files
VIZ_DIR    = Path("path/to/hovernet_output/visualizations")
# ───────────────────────────────────────────────────────

# Create visualization directory
VIZ_DIR.mkdir(exist_ok=True)

# CORRECTED Color dictionary for nucleus types (matching PanNuke model)
# Using BGR values directly for OpenCV to avoid conversion issues
COLOR_DICT = {
    0: ('background', (0, 0, 0)),                    # Black (shouldn't be visible)
    1: ('neoplastic_epithelial', (0, 0, 255)),      # Red 
    2: ('inflammatory', (0, 255, 255)),             # Yellow 
    3: ('connective', (0, 255, 0)),                 # Green 
    4: ('dead', (128, 0, 128)),                     # Purple 
    5: ('non_neoplastic_epithelial', (255, 0, 0)),  # Blue 
}

def visualize_tile_with_nuclei(tile_path, preds, save_path=None):
    """Visualize a tile with detected nuclei overlaid"""
    
    # Load the original tile
    tile = imread(str(tile_path))
    
    # Create overlay image
    overlay = tile.copy()
    
    # Process predictions
    if isinstance(preds, dict):
        nuclei_data = preds.values()
    elif isinstance(preds, list):
        nuclei_data = preds
    else:
        print(f"Unexpected prediction format: {type(preds)}")
        return
    
    # Count nuclei by type for verification
    type_counts = {}
    
    # Draw each nucleus
    for nucleus in nuclei_data:
        if isinstance(nucleus, dict) and "contour" in nucleus and "type" in nucleus:
            contour = nucleus["contour"]
            nuc_type = nucleus["type"]
            
            # Count by type
            type_counts[nuc_type] = type_counts.get(nuc_type, 0) + 1
            
            # Get color for this nucleus type (already in BGR)
            color_name, color_bgr = COLOR_DICT.get(nuc_type, ("unknown", (128, 128, 128)))
            
            # Draw contour (skip background type 0)
            if nuc_type != 0:
                cv2.drawContours(overlay, [contour.astype(np.int32)], -1, color_bgr, 2)
                
                # Add centroid dot
                if "centroid" in nucleus:
                    centroid = nucleus["centroid"].astype(np.int32)
                    cv2.circle(overlay, (centroid[0], centroid[1]), 3, color_bgr, -1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(tile)
    axes[0].set_title("Original Tile")
    axes[0].axis("off")
    
    # Overlay image
    axes[1].imshow(overlay)
    axes[1].set_title(f"Nuclei Detection ({len(nuclei_data)} nuclei)")
    axes[1].axis("off")
    
    # Add legend with matching colors
    legend_elements = []
    for nuc_type, (name, color_bgr) in COLOR_DICT.items():
        if nuc_type != 0:  # Skip background
            # Convert BGR to RGB for matplotlib
            color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
            count = type_counts.get(nuc_type, 0)
            legend_elements.append(plt.Line2D([0], [0], color=color_rgb, lw=2, 
                                            label=f"{name} ({count})"))
    
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    
    # Print type counts for verification
    print(f"Type counts: {type_counts}")

def create_summary_visualization():
    """Create a summary visualization showing multiple tiles"""
    
    # Get all .dat files
    dat_files = list(OUT_DIR.glob("*.dat"))
    
    if not dat_files:
        print("No .dat files found!")
        return
    
    # Create a grid of visualizations
    n_tiles = min(9, len(dat_files))  # Show max 9 tiles
    cols = 3
    rows = (n_tiles + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, dat_file in enumerate(dat_files[:n_tiles]):
        row = i // cols
        col = i % cols
        
        # Load predictions
        preds = joblib.load(dat_file)
        
        # Find corresponding tile file
        tile_files = list(TILE_DIR.glob("*.jpg"))
        if i < len(tile_files):
            tile_path = tile_files[i]
            tile = imread(str(tile_path))
            
            # Create overlay
            overlay = tile.copy()
            
            # Add nuclei
            if isinstance(preds, dict):
                nuclei_data = preds.values()
            elif isinstance(preds, list):
                nuclei_data = preds
            else:
                nuclei_data = []
            
            type_counts = {}
            for nucleus in nuclei_data:
                if isinstance(nucleus, dict) and "contour" in nucleus and "type" in nucleus:
                    contour = nucleus["contour"]
                    nuc_type = nucleus["type"]
                    type_counts[nuc_type] = type_counts.get(nuc_type, 0) + 1
                    
                    # Skip background type 0
                    if nuc_type != 0:
                        color_name, color_bgr = COLOR_DICT.get(nuc_type, ("unknown", (128, 128, 128)))
                        cv2.drawContours(overlay, [contour.astype(np.int32)], -1, color_bgr, 1)
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"Tile {i+1}: {len(nuclei_data)} nuclei\n"
                                   f"Neoplastic: {type_counts.get(1, 0)}, "
                                   f"Dead: {type_counts.get(4, 0)}")
            axes[row, col].axis("off")
        else:
            axes[row, col].text(0.5, 0.5, "No tile found", ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f"Tile {i+1}")
            axes[row, col].axis("off")
    
    # Hide empty subplots
    for i in range(n_tiles, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "summary_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved summary visualization to: {VIZ_DIR / 'summary_visualization.png'}")

# Main execution
if __name__ == "__main__":
    print("Creating nucleus segmentation visualizations...")
    
    # Get all .dat files
    dat_files = list(OUT_DIR.glob("*.dat"))
    tile_files = list(TILE_DIR.glob("*.jpg"))
    
    print(f"Found {len(dat_files)} prediction files and {len(tile_files)} tile files")
    
    # Create individual visualizations for first few tiles
    for i, dat_file in enumerate(dat_files[:3]):  # Show first 3 tiles
        print(f"\nProcessing tile {i+1}...")
        
        # Load predictions
        preds = joblib.load(dat_file)
        
        # Find corresponding tile
        if i < len(tile_files):
            tile_path = tile_files[i]
            save_path = VIZ_DIR / f"tile_{i+1}_nuclei.png"
            visualize_tile_with_nuclei(tile_path, preds, save_path)
        else:
            print(f"No tile file found for prediction {i+1}")
    
    # Create summary visualization
    print("\nCreating summary visualization...")
    create_summary_visualization()
    
    print(f"\nAll visualizations saved to: {VIZ_DIR}")