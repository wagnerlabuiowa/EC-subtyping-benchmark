#!/usr/bin/env python
"""
Generate:
  • violin + box plots for HoVerNet morphometric metrics
  • combined violin + jitter plot for cell‑type composition
"""

# ── IMPORTS ───────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")     # nicer theme

# ── CONFIG — update these paths for your environment ──────────────────
xlsx_path = Path("path/to/HoVerNet.xlsx")   # Excel file with per-tile HoVerNet morphometrics
out_dir   = Path("plots")
out_dir.mkdir(exist_ok=True)

metrics = [
    "mean_area",
    "cv_area",
    "PI",
    "cv_ecc",
    "cv_circ",
]
cell_types = [
    "Neoplastic_epithelial", 
    "Inflammatory",
    "Connective", 
    "Dead", 
    "Non-neoplastic_epithelial",
]

# ── LOAD & PREP ───────────────────────────────────────────────────────
sheets = pd.read_excel(xlsx_path, sheet_name=None)
data   = pd.concat(
            [df.assign(subtype=name) for name, df in sheets.items()],
            ignore_index=True
         )

subtypes = sorted(data["subtype"].unique())

# Add per‑tile cell‑type proportions
prop_df = data.copy()
prop_df["total"] = prop_df[cell_types].sum(axis=1)
prop_df[cell_types] = prop_df[cell_types].div(prop_df["total"], axis=0)
melt = prop_df.melt(
    id_vars="subtype", value_vars=cell_types,
    var_name="cell_type", value_name="prop"
)

# ── 1. Metric violin + box plots ─────────────────────────────────────
for metric in metrics:
    grouped = [data.loc[data.subtype == st, metric].dropna()
               for st in subtypes]

    # Violin
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.violinplot(grouped, showmeans=True, showextrema=True, showmedians=True)
    ax.set_xticks(range(1, len(subtypes) + 1))
    ax.set_xticklabels(subtypes, rotation=45)
    ax.set_ylabel(metric)
    ax.set_title(f"Violin plot of {metric} by subtype")
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_violin.png", dpi=300)
    plt.close(fig)

    # Box
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(grouped, showmeans=True,
               meanprops=dict(marker="^", markerfacecolor="red",
                              markersize=6))
    ax.set_xticks(range(1, len(subtypes) + 1))
    ax.set_xticklabels(subtypes, rotation=45)
    ax.set_ylabel(metric)
    ax.set_title(f"Box plot of {metric} by subtype")
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_box.png", dpi=300)
    plt.close(fig)

# ── 2. Cell‑type composition: violin + jitter, one plot per cell type ─────────────
# Custom color palette for subtypes (pastel versions)
subtype_colors = {
    "CNV-H": "lightcoral",  # pastel orange
    "CNV-L": "lightblue",   # pastel blue
    "MSI-H": "lightgreen",  # pastel green
    "POLE": "plum"          # pastel purple
}

# Use consistent y-axis range for proportions (0 to 1)
y_min, y_max = 0, 1

for cell_type in cell_types:
    fig, ax = plt.subplots(figsize=(7, 5))
    data_ct = melt[melt["cell_type"] == cell_type]
    
    # Create color list in the same order as subtypes
    colors = [subtype_colors.get(st, "gray") for st in subtypes]
    
    sns.violinplot(
        data=data_ct, x="subtype", y="prop",
        order=subtypes,
        inner=None, cut=0, palette=colors, ax=ax, width=0.8
    )
    sns.stripplot(
        data=data_ct, x="subtype", y="prop",
        order=subtypes,
        dodge=False, jitter=True, marker=".", size=3,
        linewidth=0, color="k", ax=ax
    )
    
    # Set consistent y-axis limits for proportions
    ax.set_ylim(y_min, y_max)
    
    ax.set_ylabel("Proportion of Nuclei per Tile")
    ax.set_xlabel("Subtype")
    ax.set_title(f"Tile-level {cell_type} Composition by Subtype")
    fig.tight_layout()
    fig.savefig(out_dir / f"celltype_{cell_type.replace(' ', '_')}_violin_jitter.png", dpi=300)
    plt.close(fig)

print(f"✅ All figures saved to: {out_dir.resolve()}")
