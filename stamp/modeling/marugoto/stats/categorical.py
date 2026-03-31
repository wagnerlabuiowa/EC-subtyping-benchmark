#!/usr/bin/env python3
"""Calculate statistics for deployments on categorical targets."""

from pathlib import Path
import pandas as pd
from sklearn import metrics
import scipy.stats as st
import numpy as np

__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'mvantreeck@ukaachen.de'

__all__ = ['categorical', 'aggregate_categorical_stats',
           'categorical_aggregated_']

score_labels = ['roc_auc_score', 'average_precision_score', 'p_value', 'count',
                'accuracy', 'precision', 'recall', 'specificity', 'f1_score']

def compute_specificity(y_true, y_pred):
    """Compute specificity (true negative rate) for binary classification."""
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def compute_micro_metrics(y_true, y_pred):
    """Compute micro-averaged precision, recall, and F1-score."""
    precision = metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    return precision, recall, f1

def compute_multiclass_auc(y_true_bin, y_pred_probs):
    """Compute overall multiclass AUC using one-vs-rest strategy."""
    return metrics.roc_auc_score(y_true_bin, y_pred_probs, average='macro', multi_class='ovr')

def categorical(preds_df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Calculates stats for categorical prediction tables."""
    categories = preds_df[target_label].unique()
    y_true = preds_df[target_label]
    y_pred_probs = preds_df[[f'{target_label}_{cat}' for cat in categories]].astype(float).values
    y_pred = y_pred_probs.argmax(axis=1)
    y_true_bin = np.eye(len(categories))[np.array([categories.tolist().index(y) for y in y_true])]

    stats_df = pd.DataFrame(index=categories)

    # Class counts
    stats_df['count'] = [sum(y_true == cat) for cat in categories]

    # Metrics
    stats_df['roc_auc_score'] = [
        metrics.roc_auc_score((y_true == cat).astype(int), y_pred_probs[:, i])
        for i, cat in enumerate(categories)
    ]
    stats_df['average_precision_score'] = [
        metrics.average_precision_score((y_true == cat).astype(int), y_pred_probs[:, i])
        for i, cat in enumerate(categories)
    ]
    stats_df['p_value'] = [
        st.ttest_ind(y_pred_probs[:, i][y_true == cat],
                     y_pred_probs[:, i][y_true != cat]).pvalue
        for i, cat in enumerate(categories)
    ]
    stats_df['accuracy'] = [
        metrics.accuracy_score((y_true == cat).astype(int), (y_pred == i).astype(int))
        for i, cat in enumerate(categories)
    ]
    stats_df['precision'] = [
        metrics.precision_score((y_true == cat).astype(int), (y_pred == i).astype(int), zero_division=0)
        for i, cat in enumerate(categories)
    ]
    stats_df['recall'] = [
        metrics.recall_score((y_true == cat).astype(int), (y_pred == i).astype(int), zero_division=0)
        for i, cat in enumerate(categories)
    ]
    stats_df['specificity'] = [
        compute_specificity((y_true == cat).astype(int), (y_pred == i).astype(int))
        for i, cat in enumerate(categories)
    ]
    stats_df['f1_score'] = [
        metrics.f1_score((y_true == cat).astype(int), (y_pred == i).astype(int), zero_division=0)
        for i, cat in enumerate(categories)
    ]

    # Compute macro and weighted averages
    metrics_to_average = ['roc_auc_score', 'average_precision_score', 'accuracy',
                          'precision', 'recall', 'specificity', 'f1_score']
    total_count = stats_df['count'].sum()

    macro_avg = stats_df[metrics_to_average].mean().to_dict()
    weighted_avg = {
        metric: (stats_df[metric] * stats_df['count']).sum() / total_count
        for metric in metrics_to_average
    }

    # Compute micro-averaged metrics
    micro_precision, micro_recall, micro_f1 = compute_micro_metrics(y_true_bin.argmax(axis=1), y_pred)

    # Compute multiclass AUC
    multiclass_auc = compute_multiclass_auc(y_true_bin, y_pred_probs)

    # Add macro, weighted averages, and overall metrics as additional rows
    stats_df.loc['macro_avg'] = {**macro_avg, 'count': total_count}
    stats_df.loc['weighted_avg'] = {**weighted_avg, 'count': total_count}
    stats_df.loc['micro_avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1_score': micro_f1,
        'roc_auc_score': multiclass_auc,
        'count': total_count
    }

    return stats_df

def aggregate_categorical_stats(df) -> pd.DataFrame:
    """Aggregate stats across categories and compute mean and 95% CI."""
    stats = {}
    for cat, data in df.groupby('level_1'):
        metrics_to_aggregate = ['roc_auc_score', 'average_precision_score', 'accuracy',
                                'precision', 'recall', 'specificity', 'f1_score']
        aggregated = {}
        for metric in metrics_to_aggregate:
            scores = data[metric]
            mean_val = scores.mean()
            ci_low, ci_high = st.t.interval(0.95, len(scores)-1, loc=mean_val, scale=scores.sem())
            aggregated[f'{metric}_mean'] = mean_val
            aggregated[f'{metric}_95ci_low'] = ci_low
            aggregated[f'{metric}_95ci_high'] = ci_high
        aggregated['count_sum'] = data['count'].sum()
        stats[cat] = aggregated

    return pd.DataFrame(stats).transpose()

def categorical_aggregated_(preds_csvs, outpath: str, target_label: str) -> None:
    """Calculate statistics for categorical deployments."""
    outpath = Path(outpath)
    preds_dfs = {
        Path(p).parent.name: categorical(pd.read_csv(p, dtype=str), target_label)
        for p in preds_csvs
    }
    preds_df = pd.concat(preds_dfs).sort_index()
    preds_df.to_csv(outpath / f'{target_label}-categorical-stats-individual.csv')

    stats_df = aggregate_categorical_stats(preds_df.reset_index())
    stats_df.to_csv(outpath / f'{target_label}-categorical-stats-aggregated.csv')

if __name__ == '__main__':
    from fire import Fire
    Fire(categorical_aggregated_)
