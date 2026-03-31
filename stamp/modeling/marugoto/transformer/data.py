"""Helper classes to manage pytorch data."""
from dataclasses import dataclass
import warnings
import numpy as np
from typing import Any, Iterable, Optional, Sequence, Tuple, Union, Protocol, Callable
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd


__all__ = ['BagDataset', 'make_dataset', 'get_cohort_df']


class MapDataset(Dataset):
    def __init__(
            self,
            func: Callable,
            *datasets: Sequence[Any],
            strict: bool = True
    ) -> None:
        """A dataset mapping a function over other datasets.

        Args:
            func:  Function to apply to the underlying datasets.  Has to accept
                `len(dataset)` arguments.
            datasets:  The datasets to map over.
            strict:  Enforce the datasets to have the same length.  If
                false, then all datasets will be truncated to the shortest
                dataset's length.
        """
        if strict:
            assert all(len(ds) == len(datasets[0]) for ds in datasets)
            self._len = len(datasets[0])
        elif datasets:
            self._len = min(len(ds) for ds in datasets)
        else:
            self._len = 0

        self._datasets = datasets
        self.func = func

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Any:
        return self.func(*[ds[index] for ds in self._datasets])

    def new_empty(self):
        return self


class SKLearnEncoder(Protocol):
    """An sklearn-style encoder."""
    categories_: Sequence[Sequence[str]]
    def transform(x: Sequence[Sequence[Any]]):
        ...


class EncodedDataset(MapDataset):
    def __init__(self, encode: SKLearnEncoder, values: Sequence[Any]):
        """A dataset which first encodes its input data.

        This class is useful with frameworks such as fastai, where the
        encoder is saved as part of the model.

        Args:
            encode:  an sklearn encoding to encode the data with.
            values:  data to encode.
        """
        super().__init__(self._unsqueeze_to_float32, values)
        self.encode = encode

    def _unsqueeze_to_float32(self, x):
        return torch.tensor(
            self.encode.transform(np.array(x).reshape(1, -1)),
            dtype=torch.float32)


@dataclass
class BagDataset(Dataset):
    """A dataset of bags of instances."""
    bags: Sequence[Iterable[Path]]
    """The `.h5` files containing the bags.

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x
    F, where N is the number of instances and F the number of features per
    instance.
    """
    bag_size: Optional[int] = None
    """The number of instances in each bag.

    For bags containing more instances, a random sample of `bag_size`
    instances will be drawn.  Smaller bags are padded with zeros.  If
    `bag_size` is None, all the samples will be used.
    """
    use_coords: bool = False
    """If True, read the 'coords' dataset from each .h5 file and return
    (feats, coords, length). Otherwise, return (feats, length).
    """

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index: int):
        feats_list  = []
        coords_list = []

        for bag_file in self.bags[index]:
            try:
                with h5py.File(bag_file, 'r') as f:
                    output = torch.from_numpy(f['feats'][:])

                    if self.use_coords:
                        if 'coords' not in f:
                            print(f"Warning: coords dataset not found in {bag_file}, skipping file")
                            continue
                        coords = torch.from_numpy(f['coords'][:])
                        coords_list.append(coords)

                    # If output has 3 dimensions, combine class token with
                    # average of patch tokens (e.g. Virchow)
                    if output.dim() == 3:
                        class_token = output[:, 0]
                        # Skip register tokens (indices 1-4)
                        patch_tokens = output[:, 5:]
                        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                    else:
                        embedding = output.squeeze(0)

                    feats_list.append(embedding)
            except Exception as e:
                print(f"Warning: Failed to load {bag_file}: {e}. Skipping this file.")
                continue

        if not feats_list:
            print(f"Warning: No valid features found for bag {index}. Creating dummy features.")
            dummy_feat = torch.zeros(1, 2560)
            feats_list.append(dummy_feat)

        feats = torch.cat(feats_list, dim=0).float()

        if self.use_coords:
            if not coords_list:
                print(f"Warning: No valid coordinates found for bag {index}. Creating dummy coordinates.")
                dummy_coords = torch.zeros(feats.shape[0], 2)
                coords_list.append(dummy_coords)

            coords = torch.cat(coords_list, dim=0).float()
            if coords.shape[0] != feats.shape[0]:
                print(f"Warning: feats/coords count mismatch: {feats.shape[0]} vs {coords.shape[0]}. Padding.")
                if coords.shape[0] < feats.shape[0]:
                    padding = torch.zeros(feats.shape[0] - coords.shape[0], coords.shape[1])
                    coords = torch.cat([coords, padding], dim=0)
                else:
                    coords = coords[:feats.shape[0]]

            if self.bag_size:
                feats_fixed, effective_len = _to_fixed_size_bag(feats, bag_size=self.bag_size)
                coords_fixed, _ = _to_fixed_size_bag(coords, bag_size=self.bag_size)
                return feats_fixed, coords_fixed, effective_len
            else:
                return feats, coords, len(feats)
        else:
            if self.bag_size:
                return _to_fixed_size_bag(feats, bag_size=self.bag_size)
            else:
                return feats, len(feats)


def _to_fixed_size_bag(bag: torch.Tensor, bag_size: int = 512) -> Tuple[torch.Tensor, int]:
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]

    zero_padded = torch.cat((bag_samples,
                             torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1])))
    return zero_padded, min(bag_size, len(bag))


def make_dataset(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, Sequence[Any]],
    add_features: Optional[Iterable[Tuple[Any, Sequence[Any]]]] = None,
    bag_size: Optional[int] = None,
    use_coords: bool = False,
) -> MapDataset:
    if add_features:
        return _make_multi_input_dataset(
            bags=bags, targets=targets, add_features=add_features, bag_size=bag_size)
    else:
        return _make_basic_dataset(
            bags=bags, target_enc=targets[0], targs=targets[1], bag_size=bag_size, use_coords=use_coords)


def get_target_enc(mil_learn):
    return mil_learn.dls.train.dataset._datasets[-1].encode


def _make_basic_dataset(
    *,
    bags: Sequence[Iterable[Path]],
    target_enc: SKLearnEncoder,
    targs: Sequence[Any],
    bag_size: Optional[int] = None,
    use_coords: bool = False,
) -> MapDataset:
    assert len(bags) == len(targs), \
        'number of bags and ground truths does not match!'

    bag_ds = BagDataset(bags, bag_size=bag_size, use_coords=use_coords)

    if use_coords:
        aggregator = zip_bag_targ_merge_coords
    else:
        aggregator = zip_bag_targ

    ds = MapDataset(
        aggregator,
        bag_ds,
        EncodedDataset(target_enc, targs),
    )

    return ds


def zip_bag_targ(bag, targets):
    features, lengths = bag
    return (
        features,
        lengths,
        targets.squeeze(),
    )


def zip_bag_targ_with_coords(bag, targets):
    """Zip bag with coordinates and targets (keeps coords separate)."""
    feats, coords, length = bag
    return (feats, coords, length, targets.squeeze())


def zip_bag_targ_merge_coords(bag, targets):
    """Zip bag with targets, concatenating normalised coords onto features."""
    feats, coords, length = bag
    coords_normed = normalize_coords_minmax(coords)
    feats_plus_coords = torch.cat([feats, coords_normed], dim=-1)
    return (feats_plus_coords, length, targets.squeeze())


def normalize_coords_minmax(coords):
    """Normalise coords from [min, max] to [0, 1]."""
    cmin = coords.amin(dim=0, keepdim=True)
    cmax = coords.amax(dim=0, keepdim=True)
    coord_range = (cmax - cmin).clamp_min(1e-8)
    coords_normed = (coords - cmin) / coord_range
    return coords_normed


def _make_multi_input_dataset(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, Sequence[Any]],
    add_features: Iterable[Tuple[Any, Sequence[Any]]],
    bag_size: Optional[int] = None
) -> MapDataset:
    target_enc, targs = targets
    assert len(bags) == len(targs), \
        'number of bags and ground truths does not match!'
    for i, (_, vals) in enumerate(add_features):
        assert len(vals) == len(targs), \
            f'number of additional attributes #{i} and ground truths does not match!'

    bag_ds = BagDataset(bags, bag_size=bag_size)

    add_ds = MapDataset(
        _splat_concat,
        *[
            EncodedDataset(enc, vals)
            for enc, vals in add_features
        ])

    targ_ds = EncodedDataset(target_enc, targs)

    ds = MapDataset(
        _attach_add_to_bag_and_zip_with_targ,
        bag_ds,
        add_ds,
        targ_ds,
    )

    return ds


def _splat_concat(*x): return torch.concat(x, dim=1)


def _attach_add_to_bag_and_zip_with_targ(bag, add, targ):
    return (
        torch.concat([
            bag[0],
            add.repeat(bag[0].shape[0], 1)
        ], dim=1),
        bag[1],
        targ.squeeze(),
    )


def get_cohort_df(
    clini_table: Union[Path, str], slide_table: Union[Path, str], feature_dir: Union[Path, str],
    target_label: str, categories: Iterable[str]
) -> pd.DataFrame:
    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    slide_df = pd.read_csv(slide_table, dtype=str) if Path(slide_table).suffix == '.csv' else pd.read_excel(slide_table, dtype=str)

    if 'PATIENT' not in clini_df.columns:
        raise ValueError("The PATIENT column is missing in the clini_table.\n\
                         Please ensure the patient identifier column is named PATIENT.")

    if 'PATIENT' not in slide_df.columns:
        raise ValueError("The PATIENT column is missing in the slide_table.\n\
                         Please ensure the patient identifier column is named PATIENT.")

    if 'FILENAME' in clini_df.columns and 'FILENAME' in slide_df.columns:
        clini_df = clini_df.drop(columns=['FILENAME'])

    df = clini_df.merge(slide_df, on='PATIENT')
    df = df[df[target_label].isin(categories)]

    h5s = set(feature_dir.glob('*.h5'))
    assert h5s, f'no features found in {feature_dir}!'
    h5_df = pd.DataFrame(h5s, columns=['slide_path'])
    h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem)
    df = df.merge(h5_df, on='FILENAME')

    patient_df = df.groupby('PATIENT').first().drop(columns='slide_path')
    patient_slides = df.groupby('PATIENT').slide_path.apply(list)
    df = patient_df.merge(patient_slides, left_on='PATIENT', right_index=True).reset_index()

    return df
