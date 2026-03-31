from heatmap import generate_subtype_heatmap_for_wsi_directory

# Example: generate subtype heatmap for a specific WSI
# Update paths to match your local directory structure.
results = generate_subtype_heatmap_for_wsi_directory(
    wsi_tile_dir="/path/to/tile_dir/BLOCKS/<slide_id>",
    model_path="/path/to/5Fold_Crossval/RESNET50/.../RESULTS/fold-1/bestModelFold1",
    model_name="resnet50",
    output_dir="/path/to/output/heatmaps/",
    num_classes=4,
    class_names=["CNV-H", "CNV-L", "MSI-H", "POLE"]
)