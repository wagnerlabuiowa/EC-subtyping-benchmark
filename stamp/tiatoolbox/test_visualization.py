from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tiatoolbox.utils.misc import imread

# ── CONFIG — update these paths for your environment ───
TILE_DIR   = Path("path/to/toptiles/")          # input folder with tile images
OUT_DIR    = Path("path/to/hovernet_output/")    # directory with .dat prediction files
# ───────────────────────────────────────────────────────

# Load first tile and its predictions
tile_files = list(TILE_DIR.glob("*.jpg"))
dat_files = list(OUT_DIR.glob("*.dat"))

if tile_files and dat_files:
    tile_path = tile_files[0]
    dat_file = dat_files[0]
    
    print(f"Testing visualization for:")
    print(f"  Tile: {tile_path.name}")
    print(f"  Predictions: {dat_file.name}")
    
    # Load data
    tile = imread(str(tile_path))
    preds = joblib.load(dat_file)
    
    if isinstance(preds, dict):
        nuclei_data = preds.values()
    else:
        nuclei_data = preds
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(tile)
    axes[0].set_title("Original Tile")
    axes[0].axis("off")
    
    # Overlay with all nuclei
    overlay_all = tile.copy()
    type_counts = {}
    
    for nucleus in nuclei_data:
        if isinstance(nucleus, dict) and "contour" in nucleus and "type" in nucleus:
            contour = nucleus["contour"]
            nuc_type = nucleus["type"]
            type_counts[nuc_type] = type_counts.get(nuc_type, 0) + 1
            
            # Color based on type
            if nuc_type == 0:  # background
                color = (0, 0, 0)  # black
            elif nuc_type == 1:  # neoplastic epithelial
                color = (255, 0, 0)  # red
            elif nuc_type == 4:  # dead
                color = (128, 0, 128)  # purple
            else:
                color = (128, 128, 128)  # gray
            
            cv2.drawContours(overlay_all, [contour.astype(np.int32)], -1, color, 2)
    
    axes[1].imshow(overlay_all)
    axes[1].set_title(f"All Nuclei ({len(nuclei_data)} total)")
    axes[1].axis("off")
    
    # Overlay with only neoplastic (type 1)
    overlay_neoplastic = tile.copy()
    neoplastic_count = 0
    
    for nucleus in nuclei_data:
        if isinstance(nucleus, dict) and "contour" in nucleus and "type" in nucleus:
            contour = nucleus["contour"]
            nuc_type = nucleus["type"]
            
            if nuc_type == 1:  # only neoplastic epithelial
                cv2.drawContours(overlay_neoplastic, [contour.astype(np.int32)], -1, (255, 0, 0), 2)
                neoplastic_count += 1
    
    axes[2].imshow(overlay_neoplastic)
    axes[2].set_title(f"Neoplastic Only ({neoplastic_count} nuclei)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Type counts: {type_counts}")
    print(f"Expected from CSV: 0=3, 1=38, 4=2")
    print(f"Actual detected: {type_counts}")
    
else:
    print("No tile or prediction files found!") 