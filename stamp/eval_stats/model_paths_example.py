# model_paths_example.py
#
# Template showing the expected FILEPATHS structure for evaluate_models.py.
# Each model key maps to {"train": [...5 fold dirs...], "deploy": [...5 deploy dirs...]}.
# Each directory should contain a patient-preds.csv file.
#
# Usage:
#   python evaluate_models.py --path_config model_paths_example.py --out_dir results --level patient

FILEPATHS = {
    # --- CNN baselines (from HIA pipeline) ---
    "RESNET18": {
        "train": [
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/fold-0/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/fold-1/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/fold-2/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/fold-3/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/fold-4/",
        ],
        "deploy": [
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/deploy_f0/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/deploy_f1/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/deploy_f2/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/deploy_f3/",
            "/path/to/5Fold_Crossval/RESNET18/.../RESULTS/deploy_f4/",
        ],
    },

    # --- Foundation model + TransMIL (from STAMP pipeline) ---
    "trans_UNI2": {
        "train": [
            "/path/to/output_dir/Crossval/UNI2/trans_.../fold-0/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../fold-1/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../fold-2/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../fold-3/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../fold-4/",
        ],
        "deploy": [
            "/path/to/output_dir/Crossval/UNI2/trans_.../deploy_f0/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../deploy_f1/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../deploy_f2/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../deploy_f3/",
            "/path/to/output_dir/Crossval/UNI2/trans_.../deploy_f4/",
        ],
    },

    # --- Foundation model + CLAM (from STAMP pipeline) ---
    "clam_UNI2": {
        "train": [
            "/path/to/output_dir/Crossval/UNI2/clam_.../fold-0/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../fold-1/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../fold-2/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../fold-3/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../fold-4/",
        ],
        "deploy": [
            "/path/to/output_dir/Crossval/UNI2/clam_.../deploy_f0/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../deploy_f1/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../deploy_f2/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../deploy_f3/",
            "/path/to/output_dir/Crossval/UNI2/clam_.../deploy_f4/",
        ],
    },
}
