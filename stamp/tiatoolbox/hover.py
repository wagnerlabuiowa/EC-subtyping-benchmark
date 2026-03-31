from pathlib import Path
from glob import glob
import joblib, pandas as pd
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

# ── CONFIG — update these paths for your environment ───
TILE_DIR   = Path("path/to/toptiles/")       # input folder with tile images
OUT_DIR    = Path("path/to/hovernet_output/") # where .dat files go
MODEL_NAME = "hovernet_fast-pannuke"           # pretrained EC‑friendly weights
BATCH_SIZE = 16                                # fits on 24 GB GPU
# ───────────────────────────────────────────────────────

# 1️⃣  collect image paths - only process image files
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
imgs = []
for ext in image_extensions:
    imgs.extend(glob(str(TILE_DIR / ext)))
imgs = sorted(imgs)

print(f"Found {len(imgs)} image files to process")
print("Sample files:", imgs[:3] if imgs else "No files found")

# 2️⃣  set up segmentor (removed device parameter)
segmentor = NucleusInstanceSegmentor(
    pretrained_model=MODEL_NAME,
    batch_size=BATCH_SIZE,
    num_loader_workers=4,
    num_postproc_workers=2,
)

# 3️⃣  run inference – TIAToolbox saves a .dat per tile
if imgs:
    segmentor.predict(imgs=imgs, mode="tile", save_dir=OUT_DIR)
    # (`predict` accepts any list of image paths and writes outputs to save_dir)

    # 4️⃣  post‑process: count cell types
    records = []
    TYPE_LABELS = {
        0: "background",
        1: "neoplastic_epithelial",
        2: "inflammatory",
        3: "connective",
        4: "dead",
        5: "non_neoplastic_epithelial",
    }

    for dat_file in OUT_DIR.glob("*.dat"):
        preds = joblib.load(dat_file)              # Load the predictions
        counts = {lab: 0 for lab in TYPE_LABELS.values()}

        # Check if preds is a dictionary or list and handle accordingly
        if isinstance(preds, dict):
            # If it's a dictionary (nucleus ID -> nucleus data)
            for inst in preds.values():
                t = TYPE_LABELS.get(inst["type"], "unknown")
                counts[t] += 1
        elif isinstance(preds, list):
            # If it's a list of nucleus data
            for inst in preds:
                if isinstance(inst, dict) and "type" in inst:
                    t = TYPE_LABELS.get(inst["type"], "unknown")
                    counts[t] += 1
        else:
            print(f"Warning: Unexpected prediction format in {dat_file}: {type(preds)}")
            continue

        counts["tile_name"] = dat_file.stem         # original tile filename
        records.append(counts)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUT_DIR / "tile_cell_counts.csv", index=False)
        print("Wrote", OUT_DIR / "tile_cell_counts.csv")
        print(f"Processed {len(records)} tiles")
        print("Sample counts:")
        print(df.head())
    else:
        print("No records to save - check if .dat files contain valid predictions")
else:
    print("No image files found to process!")
