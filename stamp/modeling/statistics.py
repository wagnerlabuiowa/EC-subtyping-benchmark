import sys
import argparse
from pathlib import Path
import os
from typing import Sequence
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from .marugoto.stats.categorical import categorical_aggregated_
from .marugoto.visualizations.roc import plot_multiple_decorated_roc_curves, plot_single_decorated_roc_curve
from .marugoto.visualizations.prc import plot_precision_recall_curves_, plot_single_decorated_prc_curve

def add_roc_curve_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "pred_csvs",
        metavar="PREDS_CSV",
        nargs="*",
        type=Path,
        help="Predictions to create ROC curves for.",
        default=[sys.stdin],
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        required=True,
        type=str,
        help="The target label to calculate the ROC/PRC for.",
    )
    parser.add_argument(
        "--true-class",
        metavar="CLASS",
        type=str,
        help=(
            "If provided, only this class will be treated as positive. "
            "If omitted, statistics are generated for every class that "
            "occurs in the prediction files."
        ),
    )
    parser.add_argument(
        "-o",
        "--outpath",
        metavar="PATH",
        required=True,
        type=Path,
        help=(
            "Path to save the statistics to."
        ),
    )

    parser.add_argument(
        "--n-bootstrap-samples",
        metavar="N",
        type=int,
        required=False,
        help="Number of bootstrapping samples to take for confidence interval generation.",
        default=1000
    )

    parser.add_argument(
        "--figure-width",
        metavar="INCHES",
        type=float,
        required=False,
        help="Width of the figure in inches.",
        default=3.8,
    )
    
    parser.add_argument(
        "--threshold-cmap",
        metavar="COLORMAP",
        type=plt.get_cmap,
        required=False,
        help="Draw Curve with threshold color.",
    )

    return parser


def read_table(file) -> pd.DataFrame:
    """Loads a dataframe from a file."""
    if isinstance(file, Path) and file.suffix == ".xlsx":
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)

def compute_stats(
    pred_csvs: Sequence[Path],
    target_label: str,
    output_dir: Path,
    true_class: str | None = None,
    n_bootstrap_samples: int = 1000,
    figure_width: float = 3.8,
    threshold_cmap=None,
):
    """
    Create ROC / PRC curves (and aggregated categorical statistics) for all
    classes of `target_label` or – if `true_class` is given – for that single
    class only.
    """
    # ------------------------------------------------------------------ #
    # Resolve optional arguments
    # ------------------------------------------------------------------ #
    if threshold_cmap is None:
        threshold_cmap = plt.get_cmap()
    stats_dir = output_dir / "model_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load prediction files only once
    # ------------------------------------------------------------------ #
    preds_dfs = [
        pd.read_csv(p, dtype={f"{target_label}": str, "pred": str})
        for p in pred_csvs
    ]

    # ------------------------------------------------------------------ #
    # Determine all classes we have to evaluate
    # ------------------------------------------------------------------ #
    if true_class:
        classes = [true_class]
    else:
        # Prefer the probability-column naming scheme  "<target>_<class>"
        example_df = preds_dfs[0]
        classes = [
            c[len(f"{target_label}_") :]
            for c in example_df.columns
            if c.startswith(f"{target_label}_")
        ]
        # Fall-back → use the ground-truth column
        if not classes:
            classes = sorted(example_df[target_label].unique())

    # ------------------------------------------------------------------ #
    # Loop through every class
    # ------------------------------------------------------------------ #
    for cls in classes:
        y_trues = [(df[target_label] == cls) for df in preds_dfs]
        y_preds = [pd.to_numeric(df[f"{target_label}_{cls}"]) for df in preds_dfs]

        # -------------------------------------------------------------- #
        #  Metric values (needed for the legend text)
        # -------------------------------------------------------------- #
        roc_aucs = [roc_auc_score(t, p) for t, p in zip(y_trues, y_preds)]
        prc_aucs = [average_precision_score(t, p) for t, p in zip(y_trues, y_preds)]

        # ------------------------------------------------------------------
        # produce indices that describe the helper's internal "sort-by-metric"
        # ------------------------------------------------------------------
        roc_sorted_idx = sorted(
            range(len(roc_aucs)), key=lambda i: roc_aucs[i], reverse=True
        )
        prc_sorted_idx = sorted(
            range(len(prc_aucs)), key=lambda i: prc_aucs[i], reverse=True
        )

        roc_ratio = 1.08
        # How much extra width do we need for an outside legend?
        extra_legend_space = 1.2 if len(preds_dfs) > 1 else 0.0

        # ------------------------------ ROC ---------------------------- #
        roc_fig_size = (
            figure_width + extra_legend_space,          # wider
            figure_width * roc_ratio                    # unchanged height
        )
        fig, ax = plt.subplots(figsize=roc_fig_size, dpi=300)

        if len(preds_dfs) == 1:
            plot_single_decorated_roc_curve(
                ax,
                y_trues[0],
                y_preds[0],
                title=f"{target_label} = {cls}",
                n_bootstrap_samples=n_bootstrap_samples,
                threshold_cmap=threshold_cmap,
            )
        else:
            plot_multiple_decorated_roc_curves(
                ax,
                y_trues,
                y_preds,
                title=f"{target_label} = {cls}",
                n_bootstrap_samples=None,
            )
            # -------- relabel the lines in the helper's sort order --------
            for line, i in zip(ax.get_lines()[: len(pred_csvs)], roc_sorted_idx):
                csv_path = pred_csvs[i]
                fold_name = csv_path.parent.name or csv_path.stem
                line.set_label(f"{fold_name}, AUC = {roc_aucs[i]:.2f}")

            ax.legend(frameon=False,
                      fontsize="small",
                      loc="center left",
                      bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        fig.savefig(
            stats_dir / f"AUROC_{target_label}={cls}.svg",
            bbox_inches="tight",
        )
        plt.close(fig)

        # ------------------------------ PRC ---------------------------- #
        prc_fig_size = roc_fig_size                     # keep the same size
        fig, ax = plt.subplots(figsize=prc_fig_size, dpi=300)

        if len(preds_dfs) == 1:
            plot_single_decorated_prc_curve(
                ax,
                y_trues[0],
                y_preds[0],
                title=f"{target_label} = {cls}",
                n_bootstrap_samples=n_bootstrap_samples,
            )
        else:
            plot_precision_recall_curves_(
                ax,
                pred_csvs,
                target_label=target_label,
                true_label=cls,
                outpath=stats_dir,
            )
            # -------- relabel the lines in the helper's sort order --------
            for line, i in zip(ax.get_lines()[: len(pred_csvs)], prc_sorted_idx):
                csv_path = pred_csvs[i]
                fold_name = csv_path.parent.name or csv_path.stem
                line.set_label(f"{fold_name}, AUPRC = {prc_aucs[i]:.2f}")

            ax.legend(frameon=False,
                      fontsize="small",
                      loc="center left",
                      bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        fig.savefig(
            stats_dir / f"AUPRC_{target_label}={cls}.svg",
            bbox_inches="tight",
        )
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Categorical summary (done once)
    # ------------------------------------------------------------------ #
    categorical_aggregated_(
        pred_csvs, target_label=target_label, outpath=stats_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ROC Curve.")
    args = add_roc_curve_args(parser).parse_args()
    compute_stats(pred_csvs=args.pred_csvs,
                  target_label=args.target_label,
                  true_class=args.true_class,
                  output_dir=args.outpath,
                  n_bootstrap_samples=args.n_bootstrap_samples,
                  figure_width=args.figure_width,
                  threshold_cmap=args.threshold_cmap)