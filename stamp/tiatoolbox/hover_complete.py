from pathlib import Path
from glob import glob
import joblib, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import cv2, math
from typing import Iterable, Tuple, List
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.visualization import overlay_prediction_contours
from skimage.measure import regionprops

# ── CONFIG — update these paths for your environment ──
TILE_DIR = Path("path/to/toptiles/")            # input folder with tile images
OUTPUT_BASE = Path("path/to/hovernet_output/")   # base output directory

# Auto-generate subdirectories
OUT_DIR = OUTPUT_BASE / "counts"
VIZ_DIR = OUTPUT_BASE / "visualizations"
MODEL_NAME = "hovernet_fast-pannuke"
BATCH_SIZE = 16

# ───────────────────────────────────────────────────────
COLOR_DICT = {
    0: ("Background", (255, 165, 0)),
    1: ("Neoplastic epithelial", (255, 0, 0)),
    2: ("Inflammatory", (255, 255, 0)),
    3: ("Connective", (0, 255, 0)),
    4: ("Dead", (0, 0, 0)),
    5: ("Non-neoplastic epithelial", (0, 0, 255)),
}
TYPE_LABELS = {k: v[0] for k, v in COLOR_DICT.items()}

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")

# create output folders (Segmentor will re‑create OUT_DIR if missing)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

INT_STEM = re.compile(r"^\d+$")  

def _tile_dat_files() -> list[Path]:
    """Return only the tile‑level .dat files (exclude file_map*.dat)."""
    return [p for p in OUT_DIR.glob("*.dat") if INT_STEM.match(p.stem)]

def get_tile_map():
    """Return a dict mapping tile stem to tile path for all images in TILE_DIR."""
    tile_map = {}
    for ext in IMAGE_EXTS:
        for img_path in TILE_DIR.glob(ext):
            tile_map[img_path.stem] = img_path
    return tile_map

def get_dat_map():
    """Return a dict mapping dat stem to dat path for all .dat files."""
    return {p.stem: p for p in _tile_dat_files()}

def get_matching_pairs():
    """Return sorted list of (tile_path, dat_path) pairs with matching stems."""
    tile_map = get_tile_map()
    dat_map = get_dat_map()
    common_stems = sorted(set(tile_map) & set(dat_map), key=lambda x: x)
    return [(tile_map[stem], dat_map[stem]) for stem in common_stems]

def process_tiles() -> bool:
    """Step 1: Process tiles and generate predictions"""
    print("STEP 1: Processing Tiles")
    imgs = [p for ext in IMAGE_EXTS for p in TILE_DIR.glob(ext)]
    imgs = sorted(imgs)
    print(f"Found {len(imgs)} tile images")

    if not imgs:
        print("No image files found to process!")
        return False

    # Set up segmentor
    segmentor = NucleusInstanceSegmentor(
        pretrained_model=MODEL_NAME,
        batch_size=BATCH_SIZE,
        num_loader_workers=4,
        num_postproc_workers=2,
    )

    # Run inference - let the segmentor create the directory
    print("Running nucleus segmentation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmentor.predict(
        imgs=imgs,
        mode="tile",
        save_dir=OUT_DIR,
        device=device,
        )
    print("Segmentation complete!")
    
    return True

def get_indexed_pairs():
    """Return list of (tile_path, dat_path) pairs using file_map.dat."""
    file_map_path = OUT_DIR / "file_map.dat"
    if not file_map_path.exists():
        raise RuntimeError("file_map.dat not found in output directory!")
    file_map = joblib.load(file_map_path)
    pairs = []
    if isinstance(file_map, dict):
        iterator = file_map.items()
    elif isinstance(file_map, list):
        iterator = enumerate(file_map)
    else:
        raise RuntimeError(f"Unexpected file_map type: {type(file_map)}")
    for idx, img_path in iterator:
        # If img_path is a list, use the first element
        if isinstance(img_path, list):
            if not img_path:
                print(f"Skipping index {idx}: empty image path list")
                continue
            img_path = img_path[0]
        dat_file = OUT_DIR / f"{idx}.dat"
        if dat_file.exists() and Path(img_path).exists():
            pairs.append((Path(img_path), dat_file))
        else:
            print(f"Skipping index {idx}: missing .dat or image file ({img_path})")
    return pairs

def nucleus_metrics(
    nuclei: Iterable[dict], *, mpp: float = 0.25
    ) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute per‑nucleus morphometrics for neoplastic epithelial cells (type==1).

    Returns
    -------
    areas   : list of nucleus areas in µm²
    eccs    : list of eccentricities  (0‑round … 1‑line)
    circs   : list of circularities   (0‑irreg … 1‑perfect circle)
    """
    areas, eccs, circs = [], [], []
    for nuc in nuclei:
        if nuc.get("type") != 1:
            continue                                   # tumour only

        cnt = nuc["contour"].astype(np.int32)
        if cnt.shape[0] < 5:                           # need ≥5 pts for ellipse
            continue

        # --- basic geometry (pixels) ----------------
        area_px = cv2.contourArea(cnt)
        peri_px = cv2.arcLength(cnt, True)

        # --- eccentricity via fitted ellipse --------
        # ellipse fit may fail in rare cases → try/except
        try:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        # ensure MA ≥ ma and non‑zero
        if MA < ma:
            MA, ma = ma, MA
        if MA == 0:
            continue

        # clamp ratio to [0,1] to avoid sqrt of negative
        ratio = max(0.0, min(ma / MA, 1.0))
        ecc   = math.sqrt(1.0 - ratio**2)

        # --- circularity ----------------------------
        circ = (4 * math.pi * area_px) / (peri_px**2 + 1e-9)

        # --- convert to µm² and store ---------------
        areas.append(area_px * (mpp**2))
        eccs.append(ecc)
        circs.append(circ)

    return areas, eccs, circs

def count_cells() -> bool:
    print("STEP 2: Counting Cell Types")
    pairs = get_indexed_pairs()
    records = []
    atypia_records = []
    for tile_path, dat_file in pairs:
        preds = joblib.load(dat_file)
        counts = {name: 0 for name in TYPE_LABELS.values()}
        nuclei = preds.values() if isinstance(preds, dict) else preds
        for inst in nuclei:
            if isinstance(inst, dict) and "type" in inst:
                cls_name = TYPE_LABELS.get(inst["type"], None)
                if cls_name is not None:
                    counts[cls_name] += 1
        counts["tile_name"] = tile_path.stem
        records.append(counts)

        # --- NUCLEAR ATYPIA METRICS ---
        areas, eccs, circs = nucleus_metrics(nuclei, mpp=0.25)
        if areas:  # avoid zero-division
            atypia = {
                "tile": dat_file.stem,
                "n_nuc": len(areas),
                "mean_area": np.mean(areas),
                "cv_area": np.std(areas) / np.mean(areas),
                "PI": np.percentile(areas, 90) / np.percentile(areas, 10),
                "cv_ecc": np.std(eccs) / np.mean(eccs),
                "cv_circ": np.std(circs) / np.mean(circs),
            }
            atypia_records.append(atypia)
    if not records:
        print("No .dat predictions found.")
        return False
    df = pd.DataFrame(records)
    csv_path = OUT_DIR / "tile_cell_counts.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved counts → {csv_path}")

    # --- SAVE ATYPIA METRICS ---
    if atypia_records:
        atypia_df = pd.DataFrame(atypia_records)
        atypia_csv_path = OUT_DIR / "tile_nuclear_atypia.csv"
        atypia_df.to_csv(atypia_csv_path, index=False)
        print(f"Saved nuclear atypia metrics → {atypia_csv_path}")

    return True

def visualize_tile_with_nuclei(tile_path: Path, preds, save_path: Path | None):
    tile = imread(str(tile_path))
    overlay = overlay_prediction_contours(
        canvas=tile,
        inst_dict=preds,
        type_colours=COLOR_DICT,
        line_thickness=2,
        draw_dot=False,
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(tile)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("HoVer‑Net")
    axes[1].axis("off")

    # Legend (as before)
    legend = [
        plt.Line2D([0], [0],
                   color=np.array(col)/255.,
                   lw=6, label=f"{name}")
        for i, (name, col) in COLOR_DICT.items()
    ]

    axes[1].legend(handles=legend, fontsize=8, loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ── STEP 3b: Batch visualisation & summary grid ──────────────────────
def create_visualisations() -> bool:
    print("STEP 3: Creating visualisations")
    pairs = get_indexed_pairs()
    print(f"Found {len(pairs)} matching tile/dat pairs")
    if not pairs:
        print("No predictions to visualise.")
        return False
    for i, (tile_path, dat_file) in enumerate(pairs):
        print(f"\nProcessing tile {i+1} ({tile_path.name})...")
        preds = joblib.load(dat_file)
        save_path = VIZ_DIR / f"{tile_path.stem}_nuclei.png"
        visualize_tile_with_nuclei(tile_path, preds, save_path)
    print("\nCreating summary visualization...")
    create_summary_visualization(pairs)
    print(f"\nAll visualizations saved to: {VIZ_DIR}")
    return True

def create_summary_visualization(pairs):
    n_tiles = len(pairs)
    cols = 2
    rows = (n_tiles + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    for i, (tile_path, dat_file) in enumerate(pairs[:n_tiles]):
        row = i // cols
        col = i % cols
        tile = imread(str(tile_path))
        preds = joblib.load(dat_file)
        # Use overlay_prediction_contours for overlays
        overlay = overlay_prediction_contours(
            canvas=tile,
            inst_dict=preds,
            type_colours=COLOR_DICT,
            line_thickness=2,
            draw_dot=False,
        )
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f"{tile_path.stem}")
        axes[row, col].axis("off")
    for i in range(n_tiles, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "summary_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved summary visualization to: {VIZ_DIR / 'summary_visualization.png'}")

# Main execution
if __name__ == "__main__":
    print("=== NUCLEUS SEGMENTATION PIPELINE ===")
    print(f"Processing tiles from: {TILE_DIR}")
    print(f"Output base directory: {OUTPUT_BASE}")
    print(f"  ├── Counts: {OUT_DIR}")
    print(f"  └── Visualizations: {VIZ_DIR}")
    print("=" * 50)
    
    if not process_tiles():
        raise SystemExit("No tiles found – aborting.")
    if not count_cells():
        raise SystemExit("Counting failed – aborting.")
    if not create_visualisations():
        raise SystemExit("Visualisation failed – aborting.")
    
    print("\n" + "=" * 50)
    print("=== PIPELINE COMPLETE ===")
    print(f"✅ Cell counts saved to: {OUT_DIR / 'tile_cell_counts.csv'}")
    print(f"✅ Visualizations saved to: {VIZ_DIR}")
    print("=" * 50) 