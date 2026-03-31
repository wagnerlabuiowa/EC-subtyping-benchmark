from datetime import datetime
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Dict
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fastai.vision.learner import load_learner
import torch
import torch.nn as nn

from .base import train, deploy
from .data import get_cohort_df, get_target_enc, SKLearnEncoder

__all__ = [
    'train_categorical_model_', 'deploy_categorical_model_', 'categorical_crossval_']


PathLike = Union[str, Path]


class IncompatibleVersionError(Exception):
    """Exception raised for loading a model with an incompatible version."""
    pass


def safe_load_learner(model_path, use_cpu):    
    try:
        learn = load_learner(model_path, cpu=use_cpu) # if False will use GPU instead
        return learn
    except ModuleNotFoundError as e:
        if e.name == "stamp.modeling.marugoto.transformer.ViT":
            raise IncompatibleVersionError(
                "The model checkpoint is incompatible with the current version of STAMP (>= 1.1.0). "
                "Please use STAMP version <= 1.0.3 to deploy this checkpoint."
            ) from e
        else:
            raise


def _resolve_device(device: Optional[Union[int, str]]) -> torch.device:
    """
    Convert a user-supplied *device* argument into a torch.device.

    Accepted inputs
    ----------------------------------------------------------------
        None            → cuda (if available) else cpu
        0 / "0"         → cuda:0
        "cuda:1"        → cuda:1
        "cpu"           → cpu
    """
    if device is None:
        print("Using GPU if available, else CPU")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, int):
        print(f"Using GPU {device}")
        return torch.device(f"cuda:{device}")

    device = str(device).lower()
    if device == "cpu":
        print("Using CPU")
        return torch.device("cpu")
    if device.startswith("cuda"):
        print(f"Using GPU {device}")
        return torch.device(device)           # "cuda" / "cuda:1"
    # bare number as string  → assume GPU index
    return torch.device(f"cuda:{device}")


def train_categorical_model_(
    clini_table: PathLike,
    slide_table: PathLike,
    feature_dir: PathLike,
    output_path: PathLike,
    *,
    target_label: str,
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    categories: Optional[Iterable[str]] = None,
    loss_type: str = 'ce',
    checkpoint: Optional[Path] = None,
    model_type: str = 'transmil',
    use_coords: bool = False,
    bag_size: int = 512,
    device: Optional[Union[int, str]] = None,      # ← NEW
) -> None:
    """Train a categorical model on a cohort's tile's features.

    Args:
        clini_table:  Path to the clini table.
        slide_table:  Path to the slide tabel.
        target_label:  Label to train for.
        categories:  Categories to train for, or all categories appearing in the
            clini table if none given (e.g. '["MSIH", "nonMSIH"]').
        feature_dir:  Path containing the features.
        output_path:  File to save model in.
    """
    device = _resolve_device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")

    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    from datetime import datetime
    info = {
        'description': 'MIL training',
        'clini': str(Path(clini_table).absolute()),
        'slide': str(Path(slide_table).absolute()),
        'feature_dir': str(feature_dir.absolute()),
        'target_label': str(target_label),
        'cat_labels': [str(c) for c in cat_labels],
        'cont_labels': [str(c) for c in cont_labels],
        'model_type': str(model_type),
        'loss_type': str(loss_type),
        'use_coords': str(use_coords),
        'bag_size': bag_size,
        'checkpoint': str(checkpoint),
        'output_path': str(output_path.absolute()),
        'datetime': datetime.now().astimezone().isoformat()}

    model_path = output_path/'export.pkl'
    if model_path.exists():
        print(f'{model_path} already exists. Skipping...')
        return

    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    slide_df = pd.read_csv(slide_table, dtype=str) if Path(slide_table).suffix == '.csv' else pd.read_excel(slide_table, dtype=str)
 
    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    # TODO move into get_cohort_df
    if not categories:
        categories = df[target_label].unique()
    categories = np.array(categories)
    info['categories'] = list(categories)

    df = get_cohort_df(clini_table, slide_table, feature_dir, target_label, categories)

    print('Overall distribution')
    print(df[target_label].value_counts())
    info['class distribution'] = {'overall': {  # type: ignore
        k: int(v) for k, v in df[target_label].value_counts().items()}}

    # Split off validation set
    train_patients, valid_patients = train_test_split(df.PATIENT, stratify=df[target_label], random_state=1337)
    train_df = df[df.PATIENT.isin(train_patients)]
    valid_df = df[df.PATIENT.isin(valid_patients)]
    train_df.drop(columns='slide_path').to_csv(output_path/'train.csv', index=False)
    valid_df.drop(columns='slide_path').to_csv(output_path/'valid.csv', index=False)

    info['class distribution']['training'] = {      # type: ignore
        k: int(v) for k, v in train_df[target_label].value_counts().items()}
    info['class distribution']['validation'] = {    # type: ignore
        k: int(v) for k, v in valid_df[target_label].value_counts().items()}

    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)

    target_enc = OneHotEncoder(sparse_output=False).fit(categories.reshape(-1, 1))

    add_features = []
    if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), df[cat_labels].values))
    if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), df[cont_labels].values))

    learn = train(
        bags=df.slide_path.values,
        targets=(target_enc, df[target_label].values),
        add_features=add_features,
        valid_idxs=df.PATIENT.isin(valid_patients).values,
        path=output_path,
        cores= 4, #max(1, os.cpu_count() // 4),
        model_type=model_type,
        loss_type=loss_type,
        checkpoint=checkpoint,
        use_coords=use_coords,
        bag_size=bag_size
    )

    # save some additional information to the learner to make deployment easier
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    learn.export()

    # NEW 1️⃣ – add the pretty-printed model to the info-dict
    info['model_architecture'] = str(learn.model)

    # rewrite the info.json so the new key is persisted
    with (output_path / 'info.json').open('w') as f:
        json.dump(info, f, indent=2)

    # info['models'].append({
    #     'fold'       : fold,
    #     'path'       : str(export_path.resolve()),
    #     'architecture': str(learn.model)
    # })


def _make_cat_enc(df, cats) -> SKLearnEncoder:
    # create a scaled one-hot encoder for the categorical values
    #
    # due to weirdeties in sklearn's OneHotEncoder.fit we fill NAs with other values
    # randomly sampled with the same probability as their distribution in the
    # dataset.  This is necessary for correctly determining StandardScaler's weigth
    fitting_cats = []
    for cat in cats:
        weights = df[cat].value_counts(normalize=True)
        non_na_samples = df[cat].fillna(pd.Series(np.random.choice(weights.index, len(df), p=weights)))
        fitting_cats.append(non_na_samples)
    cat_samples = np.stack(fitting_cats, axis=1)
    cat_enc = make_pipeline(
        OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
        StandardScaler(),
    ).fit(cat_samples)
    return cat_enc


def _make_cont_enc(df, conts) -> SKLearnEncoder:
    cont_enc = make_pipeline(
        StandardScaler(),
        SimpleImputer(fill_value=0)
    ).fit(df[conts].values)
    return cont_enc


def deploy_categorical_model_(
    clini_table: PathLike,
    slide_table: PathLike,
    feature_dir: PathLike,
    model_path: PathLike,
    output_path: PathLike,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[str] = None,
    cont_labels: Optional[str] = None,
    use_coords: bool = False,
    device: Optional[Union[int, str]] = None,      # ← NEW
) -> None:
    """Deploy a categorical model on a cohort's tile's features.

    Args:
        clini_table:  Path to the clini table.
        slide_table:  Path to the slide tabel.
        target_label:  Label to train for.
        feature_dir:  Path containing the features.
        model_path:  Path of the model to deploy.
        output_path:  File to save model in.
    """
    device = _resolve_device(device)
    if device.type == "cuda" and device.index is not None:
        torch.set_float32_matmul_precision("high")
    else:
        use_cpu = (device.type == "cpu")
        # allow for usage of TensorFloat32 as internal dtype for matmul on modern NVIDIA GPUs
        torch.set_float32_matmul_precision("high")
    
    use_cpu= (device.type == "cpu") # True or False
    
    feature_dir = Path(feature_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)
    if (preds_csv := output_path/'patient-preds.csv').exists():
        print(f'{preds_csv} already exists!  Skipping...')
        return


    learn = safe_load_learner(model_path, use_cpu=use_cpu)
    target_enc = get_target_enc(learn)
    categories = target_enc.categories_[0]

    target_label = target_label or learn.target_label
    cat_labels = cat_labels or learn.cat_labels
    cont_labels = cont_labels or learn.cont_labels

    test_df = get_cohort_df(clini_table, slide_table, feature_dir, target_label, categories)

    patient_preds_df = deploy(
        test_df=test_df, 
        learn=learn, 
        target_label=target_label, 
        device=device,
        use_coords=use_coords
        )
    output_path.mkdir(parents=True, exist_ok=True)
    patient_preds_df.to_csv(preds_csv, index=False)


def categorical_crossval_(
    clini_table: PathLike, slide_table: PathLike, feature_dir: PathLike, output_path: PathLike,
    *,
    target_label: str,
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    loss_type: str = 'ce',
    model_type: str = 'transmil',
    n_splits: int = 5,
    use_coords: bool = False,
    categories: Optional[Iterable[str]] = None,
    checkpoint: Optional[Path] = None,
    bag_size: int = 512,
    device: Optional[Union[int, str]] = None,      # ← NEW
) -> None:
    """Performs a cross-validation for a categorical target.

    Args:
        clini_table:  Path to the clini table.
        slide_table:  Path to the slide tabel.
        feature_dir:  Path containing the features.
        target_label:  Label to train for.
        output_path:  File to save model and the results in.
        n_splits:  The number of folds.
        loss_type:  The loss type to use.
        model_type:  The model type to use.
        categories:  Categories to train for, or all categories appearing in the
            clini table if none given (e.g. '["MSIH", "nonMSIH"]').
    """
    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    device = _resolve_device(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")

    use_cpu = device.type == "cpu"

    # big fat bookkeeping object
    info = {
        'description': 'MIL cross-validation',
        'clini': str(Path(clini_table).absolute()),
        'slide': str(Path(slide_table).absolute()),
        'feature_dir': str(feature_dir.absolute()),
        'target_label': str(target_label),
        'cat_labels': [str(c) for c in cat_labels],
        'cont_labels': [str(c) for c in cont_labels],
        'loss_type': str(loss_type),
        'model_type': str(model_type),
        'use_coords': str(use_coords),
        'output_path': str(output_path.absolute()),
        'n_splits': n_splits,
        'bag_size': bag_size,
        'datetime': datetime.now().astimezone().isoformat(),
        'device': str(device)
    }

    info['models'] = []                             # ←  NEW ❶

    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    slide_df = pd.read_csv(slide_table, dtype=str) if Path(slide_table).suffix == '.csv' else pd.read_excel(slide_table, dtype=str)


    if 'PATIENT' not in clini_df.columns:
        raise ValueError("The PATIENT column is missing in the clini_table.\n\
                         Please ensure the patient identifier column is named PATIENT.")
    
    if 'PATIENT' not in slide_df.columns:
        raise ValueError("The PATIENT column is missing in the slide_table.\n\
                         Please ensure the patient identifier column is named PATIENT.")

    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    if not categories:
        categories = df[target_label].unique()
    categories = np.array(categories)
    info['categories'] = list(categories)

    df = get_cohort_df(clini_table, slide_table, feature_dir, target_label, categories)

    info['class distribution'] = {'overall': {
        k: int(v) for k, v in df[target_label].value_counts().items()}}

    # Extract true patient ID (part before underscore) for patient-level splitting
    # PATIENT column contains slide IDs like "C3L-00098_01", true patient is "C3L-00098"
    df['TRUE_PATIENT_ID'] = df['PATIENT'].str.split('_').str[0]

    target_enc = OneHotEncoder(sparse_output=False).fit(categories.reshape(-1, 1))

    if (fold_path := output_path/'folds.pt').exists():
        folds = torch.load(fold_path)
    else:
        #added shuffling with seed 1337
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
        # Group by TRUE_PATIENT_ID for splitting, but keep all slides
        patient_df = df.groupby('TRUE_PATIENT_ID').first().reset_index()
        folds = tuple(skf.split(patient_df.TRUE_PATIENT_ID, patient_df[target_label]))
        torch.save(folds, fold_path)

    info['folds'] = [
        {
            part: list(df.PATIENT[df.TRUE_PATIENT_ID.isin(
                patient_df.TRUE_PATIENT_ID.iloc[folds[fold][i]].values
            )])
            for i, part in enumerate(['train', 'test'])
        }
        for fold in range(n_splits) ]

    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)

    for fold, (train_idxs, test_idxs) in enumerate(folds):
        print(f"\nFold: {fold+1}/{n_splits}")
        fold_path = output_path/f'fold-{fold}'
        if (preds_csv := fold_path/'patient-preds.csv').exists():
            print(f'{preds_csv} already exists!  Skipping...')
            continue
        elif (fold_path/'export.pkl').exists():
            learn = safe_load_learner(fold_path/'export.pkl', use_cpu=use_cpu)
        else:
            # Get true patient IDs for train/test split
            train_patient_ids = patient_df.TRUE_PATIENT_ID.iloc[train_idxs].values
            test_patient_ids = patient_df.TRUE_PATIENT_ID.iloc[test_idxs].values
            
            # Get all rows (slides) for these patients
            fold_train_df = df[df.TRUE_PATIENT_ID.isin(train_patient_ids)]
            fold_test_df = df[df.TRUE_PATIENT_ID.isin(test_patient_ids)]
            
            learn = _crossval_train(
                fold_path=fold_path,
                fold_df=fold_train_df,
                fold=fold,
                info=info,
                target_label=target_label,
                target_enc=target_enc,
                cat_labels=cat_labels,
                cont_labels=cont_labels,
                loss_type=loss_type,
                model_type=model_type,
                use_coords=use_coords,
                bag_size=bag_size
            )
            learn.export()

        fold_test_df = df[df.TRUE_PATIENT_ID.isin(test_patient_ids)]
        fold_test_df.drop(columns='slide_path').to_csv(fold_path/'test.csv', index=False)
        patient_preds_df = deploy(
            test_df=fold_test_df, learn=learn,
            target_label=target_label, cat_labels=cat_labels, 
            cont_labels=cont_labels, device=device, use_coords=use_coords)
        patient_preds_df.to_csv(preds_csv, index=False)

        info['models'].append({
            'fold'       : fold,
            'path'       : str(fold_path/'export.pkl'),
            'architecture': str(learn.model)
        })


def _crossval_train(
    *, fold_path, fold_df, fold, info, target_label, target_enc, cat_labels, cont_labels, loss_type, model_type, use_coords, bag_size
):
    """Helper function for training the folds."""
    # fold_df now contains all slides for the train patients
    fold_path.mkdir(exist_ok=True, parents=True)

    info['class distribution'][f'fold {fold}'] = {'overall': {
        k: int(v) for k, v in fold_df[target_label].value_counts().items()}}

    # Split train/valid at patient level (not slide level)
    # Extract true patient IDs for stratification
    if 'TRUE_PATIENT_ID' not in fold_df.columns:
        fold_df['TRUE_PATIENT_ID'] = fold_df['PATIENT'].str.split('_').str[0]
    
    patient_df = fold_df.groupby('TRUE_PATIENT_ID').first().reset_index()
    train_patients, valid_patients = train_test_split(
        patient_df.TRUE_PATIENT_ID, stratify=patient_df[target_label], random_state=1337)
    
    # Get all slides for these patients
    train_df = fold_df[fold_df.TRUE_PATIENT_ID.isin(train_patients)]
    valid_df = fold_df[fold_df.TRUE_PATIENT_ID.isin(valid_patients)]
    train_df.drop(columns='slide_path').to_csv(fold_path/'train.csv', index=False)
    valid_df.drop(columns='slide_path').to_csv(fold_path/'valid.csv', index=False)

    info['class distribution'][f'fold {fold}']['training'] = {
        k: int(v) for k, v in train_df[target_label].value_counts().items()}
    info['class distribution'][f'fold {fold}']['validation'] = {
        k: int(v) for k, v in valid_df[target_label].value_counts().items()}

    add_features = []
    if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), fold_df[cat_labels].values))
    if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), fold_df[cont_labels].values))

    # valid_idxs should be True for all slides from validation patients
    valid_idxs = fold_df.TRUE_PATIENT_ID.isin(valid_patients).values

    learn = train(
        bags=fold_df.slide_path.values,
        targets=(target_enc, fold_df[target_label].values),
        add_features=add_features,
        valid_idxs=valid_idxs,
        path=fold_path,
        cores= 4, #max(1, os.cpu_count() // 4),
        loss_type=loss_type,
        model_type=model_type,
        use_coords=use_coords,
        #checkpoint=checkpoint
        bag_size=bag_size
    )
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    return learn

