from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# 1️⃣  Load the pretrained PanNuke model using the segmentor
try:
    segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=4,
        num_loader_workers=2,
        num_postproc_workers=2,
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# 2️⃣  Read the tile — update to your tile path
tile_path = "path/to/toptiles/sample_tile.jpg"

try:
    tile = imread(tile_path)
    print(f"✅ Image loaded successfully, shape: {tile.shape}")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    exit(1)

# 3️⃣  Clean up existing output directory
output_dir = "hovernet_smoketest_output"
if os.path.exists(output_dir):
    print(f"🗑️  Removing existing output directory: {output_dir}")
    shutil.rmtree(output_dir)

# 4️⃣  Run inference using the correct parameters
try:
    # Remove unsupported parameters and use the correct approach
    tile_output = segmentor.predict(
        [tile_path],  # Pass as a list of file paths
        save_dir=output_dir,
        mode="tile",
        crash_on_exception=True,
    )
    print("✅ Inference completed successfully")
    print(f"Result type: {type(tile_output)}")
    print(f"Result: {tile_output}")
except Exception as e:
    print(f"❌ Error during inference: {e}")
    exit(1)

# 5️⃣  Load and visualize results using the documentation approach
try:
    # Load the results from the saved files
    import joblib
    
    # The result should be a list of tuples (input_path, output_path)
    if isinstance(tile_output, list) and len(tile_output) > 0:
        input_path, output_path = tile_output[0]
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        
        # Load the prediction results
        wsi_pred = joblib.load(f"{output_path}.dat")
        print(f'Number of detected nuclei: {len(wsi_pred)}')
        
        if len(wsi_pred) > 0:
            # Extract nucleus IDs and select a random nucleus
            nuc_id_list = list(wsi_pred.keys())
            selected_nuc_id = nuc_id_list[0]  # Use first nucleus
            print(f'Nucleus prediction structure for nucleus ID: {selected_nuc_id}')
            sample_nuc = wsi_pred[selected_nuc_id]
            print(f"Sample nucleus keys: {sample_nuc.keys()}")
            print(f'Bounding box: {sample_nuc["box"]}')
            print(f'Centroid: {sample_nuc["centroid"]}')
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            # Original image
            plt.subplot(2, 3, 1)
            plt.imshow(tile)
            plt.title("Original Image")
            plt.axis("off")
            
            # Show individual nuclei with contours
            import cv2
            bb = 128  # box size for patch extraction around each nucleus
            
            # Color dictionary for nucleus types
            color_dict = {0: ('neoplastic epithelial', (255, 0, 0)),
                          1: ('Inflammatory', (255, 255, 0)),
                          2: ('Connective', (0, 255, 0)),
                          3: ('Dead', (0, 0, 0)),
                          4: ('non-neoplastic epithelial', (0, 0, 255))}
            
            # Show first 4 nuclei
            for i in range(min(4, len(wsi_pred))):
                selected_nuc_id = nuc_id_list[i]
                sample_nuc = wsi_pred[selected_nuc_id]
                cent = np.int32(sample_nuc['centroid'])
                contour = sample_nuc['contour']
                
                # Create a patch around the nucleus
                y_start = max(0, cent[1] - bb//2)
                y_end = min(tile.shape[0], cent[1] + bb//2)
                x_start = max(0, cent[0] - bb//2)
                x_end = min(tile.shape[1], cent[0] + bb//2)
                
                nuc_patch = tile[y_start:y_end, x_start:x_end]
                
                # Adjust contour coordinates to patch coordinates
                contour_adjusted = contour.copy()
                contour_adjusted[:, 0] -= x_start
                contour_adjusted[:, 1] -= y_start
                
                # Overlay contour on the patch
                overlaid_patch = cv2.drawContours(nuc_patch.copy(), [contour_adjusted.astype(np.int32)], -1, (255, 255, 0), 2)
                
                plt.subplot(2, 4, i + 5)
                plt.imshow(overlaid_patch)
                plt.title(f"{color_dict[sample_nuc['type']][0]}\nCentroid: {cent}")
                plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            print("✅ Visualization completed")
        else:
            print("❌ No nuclei detected")
    else:
        print("❌ Unexpected result format")
        
except Exception as e:
    print(f"❌ Error during visualization: {e}")
    print(f"Result type: {type(tile_output)}")
    if isinstance(tile_output, list):
        print(f"Result length: {len(tile_output)}")
    exit(1)
