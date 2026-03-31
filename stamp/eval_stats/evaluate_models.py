#!/usr/bin/env python

"""
Evaluate CV-fold and external-deploy predictions.

Usage
-----
python evaluate_models.py \
    --pred_dir predictions \
    --out_dir results \
    --level slide      # or "patient"
"""

import argparse, re, os, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, brier_score_loss,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from scipy import stats
from scipy.stats import ttest_rel, shapiro
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from typing import List

CLASSES = ["CNV-H", "CNV-L", "MSI-H", "POLE"]
N_BOOT = 2000      # bootstrap replications for 95 % CI & Δtests
ALPHA = 0.05

# ----------------------------------------------------------------------
def parse_filename(fname: str):
    m = re.match(r"(?P<model>[^_]+)_(?P<split>train|deploy)_fold(?P<fold>\d)\.csv", Path(fname).name)
    if not m:
        raise ValueError(f"Bad file name: {fname}")
    return m.group("model"), m.group("split"), int(m.group("fold"))

def parse_path_meta(fp: str):
    """
    Infer (model, split, fold) from a *full* file path such as
        /.../VIT/.../fold-0/patient-preds.csv     -> train fold-0
        /.../VIT/.../deploy_f3/patient-preds.csv  -> deploy fold-3
    Raises ValueError if no pattern can be recognised.
    """
    p = Path(fp)
    fold_dir = p.parent.name          # 'fold-0' | 'deploy_f3' | ...
    model    = p.parent.parent.parent.name  # three levels up -> 'VIT' etc.

    if fold_dir.startswith("fold-"):
        split = "train"
        fold  = int(fold_dir.split("-")[1])
    elif fold_dir.startswith("deploy_f"):
        split = "deploy"
        fold  = int(re.search(r"deploy_f(\d+)", fold_dir).group(1))
    else:
        raise ValueError(f"Unrecognised fold directory '{fold_dir}' in {fp}")

    return model, split, fold

# ----------------------------------------------------------------------
def load_predictions(src):
    """
    Read all prediction CSVs and append columns 'model', 'split', 'fold'.

    Parameters
    ----------
    src : str | pathlib.Path | Iterable[str]
        • directory that contains files named
          <model>_<train|deploy>_fold<k>.csv           (old behaviour)
        • or explicit list of absolute csv paths
          as produced by `gather_from_directory_map`   (new behaviour)

    Returns
    -------
    pandas.DataFrame  Concatenated predictions + metadata.
    """
    # 1) figure out which CSV files we are supposed to read
    if isinstance(src, (str, Path)):
        csv_files = glob.glob(os.path.join(src, "*.csv"))
        records = []
        for fp in csv_files:
            try:                               # a) filename encodes everything
                model, split, fold = parse_filename(fp)
            except ValueError:                 # b) else fall back to path-meta
                model, split, fold = parse_path_meta(fp)
            df = pd.read_csv(fp)
            df["model"] = model
            df["split"] = split
            df["fold"]  = fold
            records.append(df)
    else:                                # explicit list of (path, model, split, fold) tuples
        records = []
        for fp, model, split, fold in src:
            df = pd.read_csv(fp)
            df["model"] = model
            df["split"] = split
            df["fold"]  = fold
            records.append(df)

    if not records:
        raise RuntimeError("No CSV prediction files found / recognised.")

    # Concatenate all dataframes first
    df_combined = pd.concat(records, ignore_index=True)
    print("Before renaming:", df_combined.columns)

    # Rename columns in the combined dataframe
    if "Subtype" in df_combined.columns:
        print("Subtype column found, renaming to true_label")
        df_combined = df_combined.rename(columns={"Subtype": "true_label"})

    prob_rename = {f"Subtype_{c}": f"p_{c}" for c in CLASSES
                   if f"Subtype_{c}" in df_combined.columns}
    if prob_rename:
        df_combined = df_combined.rename(columns=prob_rename)

    if "PATIENT" in df_combined.columns and "patient" not in df_combined.columns:
        df_combined = df_combined.rename(columns={"PATIENT": "patient"})

    print("After renaming:", df_combined.columns)
    return df_combined

# ----------------------------------------------------------------------

def collapse_level(df: pd.DataFrame, level: str = "patient") -> pd.DataFrame:
    """
    Average multiple SLIDES that belong to the same medical *case* /
    patient.  Slides are recognised by the suffix  "…_<n>"  (e.g.
    "S14-38447_3").  The function keeps `fold` intact, so later on we may
    still compare the five deploy folds with the (optional) "ensemble"
    fold.

    Parameters
    ----------
    df     : full prediction dataframe (one row per fold & slide)
    level  : "patient" → collapse "ID_x" → "ID".  Any other value returns
             the frame untouched.

    Returns
    -------
    pandas.DataFrame  Collapsed dataframe.
    """
    if level != "patient":
        return df                    # nothing to do → slide-level frame

    prob_cols = [f"p_{c}" for c in CLASSES]

    # helper: strip trailing "DIGITS" once; keeps other underscores intact
    def base_id(x: str) -> str:
        m = re.match(r"(.+)_\d+$", x)
        return m.group(1) if m else x

    df = df.copy()
    df["base_patient"] = df["patient"].apply(base_id)

    grp = df.groupby(["model", "split", "fold", "base_patient"], sort=False)

    collapsed = (
        grp[prob_cols].mean()                 # 1) average slide-probs
           .join(grp["true_label"].first())   # 2) identical label
           .reset_index()
           .rename(columns={"base_patient": "patient"})
    )

    # optional convenience column: hard prediction after averaging
    collapsed["pred"] = (
        collapsed[prob_cols]
        .idxmax(axis=1)
        .str.replace("p_", "", regex=False)
    )

    return collapsed

# ----------------------------------------------------------------------

def paired_t_tests(df, metric, baseline=None, fdr_alpha=0.05):
    """
    Parameters
    ----------
    df : DataFrame with columns [model, fold, <metric>]
    metric : str  # e.g. "roc_auc_macro"
    baseline : str or None
        If provided, compare every model against this model first,
        then all pairwise comparisons among all models.
    """
    # pivot to models × folds
    pivot = df.pivot_table(index="model", columns="fold", values=metric)
    models = pivot.index.tolist()

    results = []
    pairs = []
    if baseline:
        pairs += [(m, baseline) for m in models if m != baseline]
    # add all pairwise comparisons among ALL models (not just foundation models)
    pairs += list(combinations(models, 2))

    for m1, m2 in pairs:
        v1, v2 = pivot.loc[m1].dropna(), pivot.loc[m2].dropna()
        if len(v1) != len(v2):
            raise ValueError(f"fold mismatch between {m1} and {m2}")
        diff = v1.values - v2.values
        # normality check (optional but reported)
        sh_p = shapiro(diff).pvalue
        t_stat, p_raw = ttest_rel(v1, v2)
        delta = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        ci_lo, ci_hi = delta - 2.776 * se, delta + 2.776 * se  # df=4
        results.append(
            dict(model_1=m1, model_2=m2, delta=delta,
                 ci_low=ci_lo, ci_high=ci_hi,
                 p_raw=p_raw, shapiro_p=sh_p))
    res_df = pd.DataFrame(results)

    # FDR correction
    res_df["p_adj"] = multipletests(res_df["p_raw"],
                                    alpha=fdr_alpha,
                                    method="fdr_bh")[1]
    return res_df

def add_ensemble_rows(df: pd.DataFrame, id_col: str = "patient") -> pd.DataFrame:
    """
    For every (model, split) pair, average probs across all folds to
    create one extra row per sample with fold='ensemble'.
    Works for either patient-level or slide-level tables.
    """
    if id_col not in df.columns:
        return df.copy()

    prob_cols = [f"p_{c}" for c in CLASSES]
    agg_frames: List[pd.DataFrame] = []

    for (model, split), sub in df.groupby(["model", "split"]):
        grp = sub.groupby(id_col)
        ens = (
            grp[prob_cols].mean()
               .join(grp["true_label"].first())
               .reset_index()
        )
        ens["model"] = model
        ens["split"] = split
        ens["fold"]  = "ensemble"
        agg_frames.append(ens)

    ensemble_df = pd.concat(agg_frames, ignore_index=True)
    return pd.concat([df, ensemble_df], ignore_index=True)

def per_sample_metrics(df):
    """Return dict of metrics for a dataframe of one model / fold / split."""
    y_true = df["true_label"].values
    y_prob = df[[f"p_{c}" for c in CLASSES]].values
    y_pred = np.array(CLASSES)[np.argmax(y_prob, axis=1)]

    out = {}

    # micro
    out["accuracy"] = accuracy_score(y_true, y_pred)

    # macro
    macro_vals = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "ap": []}
    for c_idx, cls in enumerate(CLASSES):
        y_true_bin = (y_true == cls).astype(int)
        y_prob_bin = y_prob[:, c_idx]
        y_pred_bin = (y_pred == cls).astype(int)

        macro_vals["precision"].append(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        macro_vals["recall"].append(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        macro_vals["f1"].append(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        macro_vals["roc_auc"].append(roc_auc_score(y_true_bin, y_prob_bin))
        macro_vals["ap"].append(average_precision_score(y_true_bin, y_prob_bin))

        # class-specific output
        out[f"roc_auc_{cls}"] = macro_vals["roc_auc"][-1]
        out[f"ap_{cls}"]      = macro_vals["ap"][-1]
        out[f"f1_{cls}"]      = macro_vals["f1"][-1]
        out[f"recall_{cls}"]  = macro_vals["recall"][-1]

    out["roc_auc_macro"] = np.mean(macro_vals["roc_auc"])
    out["ap_macro"]      = np.mean(macro_vals["ap"])
    out["f1_macro"]      = np.mean(macro_vals["f1"])
    out["precision_macro"]= np.mean(macro_vals["precision"])
    out["recall_macro"]  = np.mean(macro_vals["recall"])

    # calibration (Brier & slope / intercept)
    y_true_bin_all = (y_true[:, None] == np.array(CLASSES)[None, :]).astype(int).ravel()
    y_prob_all     = y_prob.ravel()
    out["brier"]   = brier_score_loss(y_true_bin_all, y_prob_all)

    # slope & intercept via logistic regression (statsmodels optional)
    try:
        import statsmodels.api as sm
        logit_mod = sm.Logit(y_true_bin_all, sm.add_constant(y_prob_all)).fit(disp=False)
        out["cal_slope"], out["cal_intercept"] = logit_mod.params[1], logit_mod.params[0]
    except Exception:
        out["cal_slope"] = np.nan
        out["cal_intercept"] = np.nan

    return out

# ----------------------------------------------------------------------
def aggregate_metrics(df):
    grouped = df.groupby(["model", "split", "fold"])
    rows = []
    for (model, split, fold), sub in tqdm(grouped, desc="Computing metrics"):
        metrics = per_sample_metrics(sub)
        metrics.update({"model": model, "split": split, "fold": fold})
        rows.append(metrics)
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
def mean_sd_ci(df_metrics, split, out_csv):
    """Compute mean ± SD and boot-strap CI for deploy."""
    metrics = df_metrics[df_metrics["split"] == split]
    no_ens  = metrics[metrics["fold"] != "ensemble"]
    
    # keep only real metric columns (numeric) – skip strings like
    # 'split' as well as the fold index
    numeric_cols = no_ens.select_dtypes(include=[np.number]).columns.tolist()
    if "fold" in numeric_cols:               # fold is merely an index, do not average it
        numeric_cols.remove("fold")

    agg = no_ens.groupby("model")[numeric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]  # flatten

    # ---------- BEST FOLD ----------
    best = no_ens.groupby("model")[["roc_auc_macro", "ap_macro"]].max() \
                 .rename(columns={"roc_auc_macro": "roc_auc_best",
                                  "ap_macro":      "ap_best"})
    agg = agg.join(best)

    # ---------- ENSEMBLE ROW ----------
    if (metrics["fold"] == "ensemble").any():
        ens = (
            metrics[metrics["fold"] == "ensemble"]
            .set_index("model")[["roc_auc_macro", "ap_macro"]]
            .rename(columns={"roc_auc_macro": "roc_auc_ensemble",
                             "ap_macro":      "ap_ensemble"})
        )
        agg = agg.join(ens)

    if split == "deploy":
        # boot-strap CI for macro ROC & AP
        ci_rows = []
        for model in metrics["model"].unique():
            sub = metrics[metrics["model"] == model]
            roc_vals = sub["roc_auc_macro"].values
            ap_vals  = sub["ap_macro"].values
            # bootstrap over folds → empirical CI
            roc_ci = np.percentile([
                np.mean(resample(roc_vals, replace=True, n_samples=len(roc_vals)))
                for _ in range(N_BOOT)
            ], [100*ALPHA/2, 100*(1-ALPHA/2)])
            ap_ci = np.percentile([
                np.mean(resample(ap_vals, replace=True, n_samples=len(ap_vals)))
                for _ in range(N_BOOT)
            ], [100*ALPHA/2, 100*(1-ALPHA/2)])

            ci_rows.append({
                "model": model,
                "roc_auc_ci_low": roc_ci[0],
                "roc_auc_ci_high": roc_ci[1],
                "ap_ci_low": ap_ci[0],
                "ap_ci_high": ap_ci[1],
            })
        ci_df = pd.DataFrame(ci_rows).set_index("model")
        agg = agg.join(ci_df)

    agg.to_csv(out_csv)
    print(f"Saved {out_csv}")
    return agg

# ----------------------------------------------------------------------
def plot_heatmap_auc(df_metrics, split, out_png):
    # keep only the per-class ROC-AUC columns → avoid averaging strings
    roc_cols = [f"roc_auc_{c}" for c in CLASSES]
    data = df_metrics[df_metrics["split"] == split]

    # either way works – we choose the explicit-column approach
    pivot = data.groupby("model")[roc_cols].mean()

    pivot.columns = CLASSES
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0.5, vmax=1.0, fmt=".2f")
    # Exclude ensemble fold when counting
    real_folds = data[data["fold"] != "ensemble"]["fold"].unique()
    plt.title(f"Per-class ROC-AUC ({split} mean over {len(real_folds)} folds)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_heatmap_prc(df_metrics, split, out_png):
    # keep only the per-class AP columns
    ap_cols = [f"ap_{c}" for c in CLASSES]
    data = df_metrics[df_metrics["split"] == split]

    pivot = data.groupby("model")[ap_cols].mean()
    pivot.columns = CLASSES
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0.5, vmax=1.0, fmt=".2f")
    # Exclude ensemble fold when counting
    real_folds = data[data["fold"] != "ensemble"]["fold"].unique()
    plt.title(f"Per-class Average Precision ({split} mean over {len(real_folds)} folds)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ----------------------------------------------------------------------
def plot_roc_pr_curves(df_pred, split, out_roc, out_pr):
    import inspect, warnings
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, RocCurveDisplay,
        PrecisionRecallDisplay
    )

    # ------------------------------------------------------------------
    # (A) ROC-CURVES ----------------------------------------------------
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for model, sub in df_pred[df_pred["split"] == split].groupby("model"):
        y_true = sub["true_label"].values
        y_prob = sub[[f"p_{c}" for c in CLASSES]].values
        # --- Newer scikit-learn (≥ 1.2) understands `multi_class`
        #     and draws a macro-average curve automatically.
        try:
            RocCurveDisplay.from_predictions(
                y_true, y_prob,
                name=model, alpha=0.8, lw=1.2, plot_chance_level=False,
                multi_class="ovr", average="macro"
            )
        # --- Older versions: fall back to manual one-vs-rest curves ----
        except (TypeError, ValueError):
            warnings.warn(
                "Your scikit-learn version does not support "
                "`multi_class` inside RocCurveDisplay. "
                "Plotting the macro-average curve only."
            )
            y_true_bin = label_binarize(y_true, classes=CLASSES)
            tpr_sum = None                       # accumulate per-class TPRs
            mean_fpr = np.linspace(0, 1, 101)
            for idx, cls in enumerate(CLASSES):
                fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
                if tpr_sum is None:
                    tpr_sum = np.interp(mean_fpr, fpr, tpr)
                else:
                    tpr_sum += np.interp(mean_fpr, fpr, tpr)
            tpr_mean = tpr_sum / len(CLASSES)
            # draw ONE macro curve -----------------------------------
            plt.plot(mean_fpr, tpr_mean, lw=1.8, alpha=0.9,
                     label=f"{model} (macro)")

    plt.title(f"ROC curves ({split} set)")
    plt.legend(fontsize="small"); plt.tight_layout()
    plt.savefig(out_roc, dpi=300); plt.close()

    # ------------------------------------------------------------------
    # (B) PRECISION–RECALL CURVES --------------------------------------
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for model, sub in df_pred[df_pred["split"] == split].groupby("model"):
        y_true = sub["true_label"].values
        y_prob = sub[[f"p_{c}" for c in CLASSES]].values
        try:
            PrecisionRecallDisplay.from_predictions(
                y_true, y_prob, name=model, alpha=0.8, lw=1.2,
                average="macro"
            )
        except (TypeError, ValueError):
            # manual macro   PR  (same idea as above)
            y_true_bin = label_binarize(y_true, classes=CLASSES)
            recall_grid = np.linspace(0, 1, 101)
            precision_sum = None
            for idx in range(len(CLASSES)):
                prec, rec, _ = precision_recall_curve(
                    y_true_bin[:, idx], y_prob[:, idx])
                if precision_sum is None:
                    precision_sum = np.interp(recall_grid, rec[::-1],
                                               prec[::-1])
                else:
                    precision_sum += np.interp(recall_grid, rec[::-1],
                                               prec[::-1])
            prec_mean = precision_sum / len(CLASSES)
            plt.plot(recall_grid, prec_mean,
                     label=f"{model} (macro)", lw=1.5, alpha=0.9)

    plt.title(f"Precision–Recall curves ({split} set)")
    plt.legend(fontsize="small"); plt.tight_layout()
    plt.savefig(out_pr, dpi=300); plt.close()


# ----------------------------------------------------------------------
def plot_confusion(model, df_pred, out_png):
    sub = df_pred[(df_pred["model"] == model) & (df_pred["split"] == "deploy")]

    y_true = sub["true_label"].values
    y_prob = sub[[f"p_{c}" for c in CLASSES]].values
    y_pred = np.array(CLASSES)[np.argmax(y_prob, axis=1)]
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES, normalize="true")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Normalised confusion matrix – {model}")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

# ----------------------------------------------------------------------
def plot_calibration(model, df_pred, out_png):
    sub = df_pred[(df_pred["model"] == model) & (df_pred["split"] == "deploy")]

    # convert multi-class probs to max-prob (for overall calibration)
    y_true = (sub["true_label"] == sub[[f"p_{c}" for c in CLASSES]]
              .idxmax(axis=1).str.replace("p_", "")).astype(int)
    y_prob = sub[[f"p_{c}" for c in CLASSES]].max(axis=1).values
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(4, 4))
    CalibrationDisplay(prob_true, prob_pred, y_prob).plot()
    plt.title(f"Reliability – {model}")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

# ----------------------------------------------------------------------
def competitive_tests(df_metrics, out_csv):
    """Bootstrap ΔROC and ΔPR-AUC between top two models (deploy macro)."""
    deploy = df_metrics[df_metrics["split"] == "deploy"]
    means = deploy.groupby("model")["roc_auc_macro"].mean()
    top2 = means.sort_values(ascending=False).head(2).index.tolist()
    m1, m2 = [deploy[deploy["model"] == m]["roc_auc_macro"].values for m in top2]
    delta = m1 - m2
    p = (np.abs(delta) < 0).mean()   # proxy; use paired bootstrap
    res = pd.DataFrame({"metric": ["roc_auc_macro"], "model1": [top2[0]], "model2": [top2[1]],
                        "delta": [delta.mean()], "p": [p]})
    res.to_csv(out_csv, index=False)

# ----------------------------------------------------------------------
def main(pred_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df_pred_raw = load_predictions(pred_dir)
    print("df_pred_raw: ", df_pred_raw.columns)

    # collapse slide→patient if requested
    df_pred_lvl = collapse_level(df_pred_raw, args.level)
    
    # append ensemble rows
    id_col = "patient" if "patient" in df_pred_lvl.columns else ("sample_id" if "sample_id" in df_pred_lvl.columns else None)
    
    df_pred = add_ensemble_rows(df_pred_lvl, id_col=id_col)

    # ------------------------------------------------------------
    # 1.  SAVE the full, row-wise prediction table for auditing
    # ------------------------------------------------------------
    compiled_csv = Path(out_dir) / "compiled_predictions.csv"
    df_pred.to_csv(compiled_csv, index=False)
    print(f"Saved full prediction table → {compiled_csv}")

    # ------------------------------------------------------------
    # THRESHOLD TEST
    # ------------------------------------------------------------
    
    if args.threshold_test:

        for cls in CLASSES:
            mask = df_pred.true_label == cls

            for t in np.linspace(0.1, 0.5, 9):
                y_pred = np.where(df_pred[f"p_{cls}"] >= t, cls,
                                df_pred[[f"p_{c}" for c in ["CNV-H","CNV-L","POLE"]]]
                                    .idxmax(axis=1).str[2:])
                recall = (y_pred[mask] == cls).mean()
                precision = (df_pred.true_label[y_pred == cls] == cls).mean()
                f1 = 2 * (recall * precision) / (recall + precision)
                print(f"{cls} thr {t:.2f}: recall {recall:.2f}, precision {precision:.2f}, f1 {f1:.2f}")
    # ------------------------------------------------------------

    df_metrics = aggregate_metrics(df_pred)

    # summary tables
    mean_sd_ci(df_metrics, "train",   Path(out_dir) / "summary_cv.csv")
    mean_sd_ci(df_metrics, "deploy",  Path(out_dir) / "summary_deploy.csv")

    # heat-maps
    plot_heatmap_auc(df_metrics, "train", Path(out_dir) / "heatmap_auc_train.png")
    plot_heatmap_auc(df_metrics, "deploy", Path(out_dir) / "heatmap_auc_deploy.png")
    plot_heatmap_prc(df_metrics, "train", Path(out_dir) / "heatmap_prc_train.png")
    plot_heatmap_prc(df_metrics, "deploy", Path(out_dir) / "heatmap_prc_deploy.png")

    # curves for both train and deploy
    plot_roc_pr_curves(df_pred, "train", Path(out_dir) / "roc_curves_train.pdf",
                       Path(out_dir) / "pr_curves_train.pdf")
    plot_roc_pr_curves(df_pred, "deploy", Path(out_dir) / "roc_curves_deploy.pdf",
                       Path(out_dir) / "pr_curves_deploy.pdf")

    # per-model plots
    for model in df_pred["model"].unique():
        plot_confusion(model, df_pred, Path(out_dir) / f"confmat_{model}.png")
        plot_calibration(model, df_pred, Path(out_dir) / f"calibration_{model}.png")

    # simple Δtest
    competitive_tests(df_metrics, Path(out_dir) / "bootstrap_tests.csv")

    # paired t-tests for statistical comparisons
    # Run for deploy split (most important for external validation)
    deploy_metrics = df_metrics[df_metrics["split"] == "deploy"]
    deploy_metrics_no_ens = deploy_metrics[deploy_metrics["fold"] != "ensemble"]
    
    # Test ROC AUC macro differences (all pairwise comparisons)
    roc_tests = paired_t_tests(deploy_metrics_no_ens, "roc_auc_macro", baseline=None)
    roc_tests.to_csv(Path(out_dir) / "paired_t_tests_roc_auc.csv", index=False)
    
    # Test AP macro differences (all pairwise comparisons)
    ap_tests = paired_t_tests(deploy_metrics_no_ens, "ap_macro", baseline=None)
    ap_tests.to_csv(Path(out_dir) / "paired_t_tests_ap.csv", index=False)
    
    print("All done!  Summaries & figures are in:", out_dir)

# ----------------------------------------------------------------------

def gather_from_directory_map(dir_map, pred_fname="patient-preds.csv"):
    """
    dir_map = {"MODEL": {"train": [...5 dirs...], "deploy": [...5 dirs...]}, ...}
    Returns a list of absolute CSV paths and lets us
    parse model/split/fold directly from the map indices.
    """
    csv_paths = []
    for model, split_dict in dir_map.items():
        for split, dir_list in split_dict.items():
            for fold, d in enumerate(sorted(dir_list)):
                full_path = Path(d) / pred_fname
                if not full_path.exists():
                    raise FileNotFoundError(full_path)
                # Store model name along with the path
                csv_paths.append((str(full_path), model, split, fold))
    return csv_paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate transformer model predictions.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--path_config", help="Python file containing FILEPATHS dict")
    group.add_argument("--csv_paths", nargs="+", help="Explicit list of CSV files")

    ap.add_argument("--pred_fname", default="patient-preds.csv",
                    help="Prediction file name inside each directory (default: patient-preds.csv)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--level", choices=["slide", "patient"], default="slide",
                help="Granularity of evaluation table (default: slide)")
    ap.add_argument("--threshold_test", default = False,
                help="Run threshold test for each class")
    args = ap.parse_args()
    

    if args.path_config:
        # Dynamically import the user-supplied Python file
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("cfg", args.path_config)
        cfg  = importlib.util.module_from_spec(spec)
        sys.modules["cfg"] = cfg
        spec.loader.exec_module(cfg)
        csv_list = gather_from_directory_map(cfg.FILEPATHS, args.pred_fname)
    else:
        csv_list = args.csv_paths

    main(csv_list, args.out_dir)


"""
---RUN---
python evaluate_models.py \
    --path_config model_paths.py \
    --out_dir results
    --level patient or slide
"""