from pathlib import Path
import joblib

# ── CONFIG — update this path for your environment ─────
OUT_DIR = Path("path/to/hovernet_output/")   # directory with .dat prediction files
# ───────────────────────────────────────────────────────

print("Analyzing actual nucleus types in your data...")

# Check first .dat file
dat_files = list(OUT_DIR.glob("*.dat"))
if dat_files:
    dat_file = dat_files[0]
    print(f"Analyzing: {dat_file.name}")
    
    preds = joblib.load(dat_file)
    
    if isinstance(preds, dict):
        nuclei_data = preds.values()
    elif isinstance(preds, list):
        nuclei_data = preds
    else:
        print(f"Unexpected format: {type(preds)}")
        exit(1)
    
    # Show first few nuclei with their actual type values
    print(f"Total nuclei: {len(nuclei_data)}")
    print("\nFirst 10 nuclei:")
    for i, nucleus in enumerate(list(nuclei_data)[:10]):
        if isinstance(nucleus, dict):
            nuc_type = nucleus.get("type", "N/A")
            centroid = nucleus.get("centroid", "N/A")
            prob = nucleus.get("prob", "N/A")
            print(f"  Nucleus {i+1}: type={nuc_type}, centroid={centroid}, prob={prob}")
    
    # Count all types
    type_counts = {}
    for nucleus in nuclei_data:
        if isinstance(nucleus, dict) and "type" in nucleus:
            nuc_type = nucleus["type"]
            type_counts[nuc_type] = type_counts.get(nuc_type, 0) + 1
    
    print(f"\nType distribution: {type_counts}") 