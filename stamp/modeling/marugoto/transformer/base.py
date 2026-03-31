from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, Sampler
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc, Callback,
    SaveModelCallback, CSVLogger, EarlyStoppingCallback,
    MixedPrecision, AMPMode, OptimWrapper
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from .data import make_dataset, SKLearnEncoder
from .TransMIL import TransMIL
from .CLAM import CLAM_SB, CLAM_MB


__all__ = ['train', 'deploy']


class FocalLoss(nn.Module):
    """A focal loss implementation for classification."""
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize FocalLoss.

        Args:
            weight: Class weights (like in nn.CrossEntropyLoss).
            alpha:  Multiplicative factor to further up-weight minority classes.
            gamma:  Focusing parameter that down-weights well-classified samples.
            reduction: 'mean', 'sum', or 'none' (see PyTorch docs).
        """
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_factor = (1.0 - pt) ** self.gamma
        loss = self.alpha * focal_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def focal_loss_multiclass(inputs, targets, alpha=1, gamma=2):
    """Multi-class focal loss implementation.

    Args:
        inputs: raw logits from the model
        targets: true class labels (integer indices or one-hot encoded)
    """
    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)

    if targets.shape == inputs.shape:
        targets_one_hot = targets.float()
    else:
        targets = targets.long()
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1]).float()

    pt = torch.sum(prob * targets_one_hot, dim=-1)
    focal_loss = -alpha * (1 - pt) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)

    return focal_loss.mean()


T = TypeVar('T')


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, np.ndarray],
    add_features: Iterable[Tuple[SKLearnEncoder, Sequence[Any]]] = [],
    valid_idxs: np.ndarray,
    n_epoch: int = 32,
    patience: int = 5,
    path: Optional[Path] = None,
    bag_size: int = 512,
    batch_size: int = 8,
    cores: int = 4,
    plot: bool = False,
    checkpoint: Optional[Path] = None,
    loss_type: str = 'ce',
    model_type: str = 'transmil',
    model_size: str = 'small',
    k_sample: int = 32,
    subtyping: bool = True,
    use_coords: bool = False,
) -> Learner:
    """Train a MIL model on bag-level image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
        model_type: Type of model to use ('transmil', 'clam_sb', or 'clam_mb')
        model_size: Size of CLAM model ('small' or 'big')
        k_sample: Number of samples for CLAM instance-level training
        subtyping: Whether to use subtyping for CLAM
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.cuda.empty_cache()

    print(f"Training config: batch_size={batch_size}, bag_size={bag_size}, "
          f"cores={cores}, model_type={model_type}, loss_type={loss_type}")

    target_enc, targs = targets
    train_ds = make_dataset(
        bags=bags[~valid_idxs],
        targets=(target_enc, targs[~valid_idxs]),
        add_features=[
            (enc, vals[~valid_idxs])
            for enc, vals in add_features],
        bag_size=bag_size,
        use_coords=use_coords
    )

    valid_ds = make_dataset(
        bags=bags[valid_idxs],
        targets=(target_enc, targs[valid_idxs]),
        add_features=[
            (enc, vals[valid_idxs])
            for enc, vals in add_features],
        bag_size=None,
        use_coords=use_coords
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=cores,
        drop_last=len(train_ds) > batch_size,
        device=device, pin_memory=device.type == "cuda"
    )

    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=cores,
        device=device, pin_memory=device.type == "cuda"
    )

    batch = train_dl.one_batch()
    feature_dim = batch[0].shape[-1]

    if model_type.lower() == 'transmil':
        model = TransMIL(
            num_classes=len(target_enc.categories_[0]), input_dim=feature_dim,
            dim=512, depth=2, heads=8, mlp_dim=512, dropout=0.0
        )
    elif model_type.lower() == 'clam_sb':
        model = CLAM_SB(
            num_classes=len(target_enc.categories_[0]),
            input_dim=feature_dim,
            size_arg=model_size,
            dropout=0.25,
            k_sample=k_sample,
            instance_loss_fn=nn.CrossEntropyLoss(),
            subtyping=subtyping
        )
    elif model_type.lower() == 'clam_mb':
        model = CLAM_MB(
            num_classes=len(target_enc.categories_[0]),
            input_dim=feature_dim,
            size_arg=model_size,
            dropout=0.25,
            k_sample=k_sample,
            instance_loss_fn=nn.CrossEntropyLoss(),
            subtyping=subtyping
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'transmil', 'clam_sb', or 'clam_mb'")

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        if 'state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['state_dict'])
        else:
            model.load_state_dict(checkpoint_data)
        print("Checkpoint loaded successfully.")

    model.to(device)
    print(f"Model: {model_type} [Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}]")

    # Class weights: inverse frequency, normalised
    counts = pd.Series(targs[~valid_idxs]).value_counts()
    weight_series = counts.sum() / counts
    weight_series /= weight_series.sum()

    weight = torch.tensor(
        list(map(weight_series.get, target_enc.categories_[0])), dtype=torch.float32, device=device)

    if loss_type == 'focal':
        alpha, gamma = 0.75, 2.0
        loss_func = FocalLoss(weight=weight, alpha=alpha, gamma=gamma)
        print(f"Using FocalLoss with alpha={alpha}, gamma={gamma}")
    elif loss_type == 'focal_multiclass':
        loss_func = focal_loss_multiclass
        print("Using focal_loss_multiclass")
    else:
        loss_func = nn.CrossEntropyLoss(weight=weight)
        print("Using standard CrossEntropyLoss")

    dls = DataLoaders(train_dl, valid_dl, device=device)

    cbs = [CSVLogger(fname=path/'history.csv')]
    metrics = []
    if len(valid_dl) != 0 and len(valid_dl.dataset) != 0:
        metrics += [RocAuc()]
        cbs += [
            SaveModelCallback(fname='model-best'),
            EarlyStoppingCallback(patience=patience),
        ]
    else:
        print("No validation set — SaveModel/EarlyStopping disabled")

    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        opt_func = partial(OptimWrapper, opt=torch.optim.AdamW),
        metrics=metrics,
        path=path
    )

    if checkpoint is not None and 'optimizer' in checkpoint_data:
        learn.opt.load_state_dict(checkpoint_data['optimizer'])

    print(f"#slides: {len(train_ds)} train, {len(valid_ds)} valid  |  "
          f"#batches/epoch: {len(train_dl)}")

    learn.fit_one_cycle(n_epoch=n_epoch, reset_opt=True, lr_max=1e-4, wd=1e-2, cbs=cbs)

    if path is not None:
        updated_checkpoint_path = path / "updated_model_checkpoint.pth"
        updated_checkpoint = {
            'state_dict': learn.model.state_dict(),
            'optimizer': learn.opt.state_dict(),
        }
        torch.save(updated_checkpoint, updated_checkpoint_path)
        print(f"Checkpoint saved to {updated_checkpoint_path}")

    if plot:
        path_plots = path / "plots"
        path_plots.mkdir(parents=True, exist_ok=True)

        learn.recorder.plot_loss()
        plt.savefig(path_plots / 'losses_plot.png')
        plt.close()

        learn.recorder.plot_sched()
        plt.savefig(path_plots / 'lr_scheduler.png')
        plt.close()

    return learn


def deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None,
    cont_labels: Optional[Sequence[str]] = None,
    device: torch.device = torch.device('cpu'),
    use_coords: bool = False
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'

    if target_label is None:
        target_label = getattr(learn, "target_label", "TARGET")
    if cat_labels is None:
        cat_labels = getattr(learn, "cat_labels", [])
    if cont_labels is None:
        cont_labels = getattr(learn, "cont_labels", [])

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None,
        use_coords=use_coords)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=1,
        device=device, pin_memory=device.type == "cuda")

    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values,
        **{f'{target_label}_{cat}': patient_preds[:, i]
            for i, cat in enumerate(categories)}})

    patient_preds = patient_preds_df[[
        f'{target_label}_{cat}' for cat in categories]].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1))
    patient_preds_df['loss'] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs),
        reduction='none')

    patient_preds_df['pred'] = categories[patient_preds.argmax(1)]

    patient_preds_df = patient_preds_df[[
        'PATIENT',
        target_label,
        'pred',
        *(f'{target_label}_{cat}' for cat in categories),
        'loss']]
    patient_preds_df = patient_preds_df.sort_values(by='loss')

    return patient_preds_df
