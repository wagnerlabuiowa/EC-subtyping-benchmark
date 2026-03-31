import argparse
from collections.abc import Collection
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from fastai.vision.learner import Learner, load_learner
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from PIL import Image
from torch import Tensor
from typing import Sequence

from stamp.preprocessing.helpers.common import supported_extensions


def load_slide_ext(wsi_dir: Path) -> openslide.OpenSlide:
    # Check if any supported extension matches the file
    if wsi_dir.suffix not in supported_extensions:
        raise FileNotFoundError(
            f"No supported slide file found for slide {wsi_dir.name} in provided directory {wsi_dir.parent}\
                                 \nOnly support for: {supported_extensions}"
        )
    else:
        return openslide.open_slide(wsi_dir)


def get_stride(coords: Tensor) -> int:
    xs = coords[:, 0].unique(sorted=True)
    stride = (xs[1:] - xs[:-1]).min()
    return stride


def gradcam_per_category(
    learn: Learner, feats: Tensor, categories: Collection
) -> tuple[Tensor, Tensor]:
    print(f"len(categories): {len(categories)}")
    print(f"categories: {categories}")
    print(f"feats shape: {feats.shape}")
    print(f"feats dim: {feats.dim()}")

    if (feats.dim() == 3):
        class_token = feats[:, 0]
        patch_tokens = feats[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1) 
        print(f"embedding shape: {embedding.shape}")
        feats_batch = embedding.expand((len(categories), *embedding.shape)).detach()
    else:
        feats_batch = feats.expand((len(categories), *feats.shape)).detach()

    feats_batch.requires_grad = True
    print(f"feats_batch shape: {feats_batch.shape}")

    preds = torch.softmax(
        learn.model(feats_batch, torch.tensor([len(feats)] * len(categories))),
        dim=1,
    )
    preds.trace().backward()
    gradcam = (feats_batch * feats_batch.grad).mean(-1).abs()
    print(f"gradcam min: {gradcam.min()}, max: {gradcam.max()}, mean: {gradcam.mean()}")    
    return preds, gradcam


def vals_to_im(
    scores: Float[Tensor, "n_tiles *d_feats"],
    norm_coords: Int[Tensor, "n_tiles *d_feats"],
) -> Float[Tensor, "i j *d_feats"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = norm_coords.max(0).values.flip(0) + 1
    im = torch.zeros((*size, *scores.shape[1:]))

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = norm_coords[:, 1] * im.shape[1] + norm_coords[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def show_thumb(slide, thumb_ax: Axes, attention: Tensor) -> None:
    DEFAULT_TIFF_MPP = 0.25
    try:
        mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Slide MPP successfully retrieved from metadata: {mpp}")
    except KeyError:
        mpp = DEFAULT_TIFF_MPP

    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])
    return np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8]


def show_class_map(
    class_ax: Axes, top_scores: Tensor, gradcam_2d, categories: Collection[str]
) -> None:
    cmap = plt.get_cmap("Pastel1")
    classes = cmap(top_scores.indices[:, :, 0])
    classes[..., -1] = (gradcam_2d.sum(-1) > 0).detach().cpu() * 1.0
    class_ax.imshow(classes)
    class_ax.legend(
    handles=[
        Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
    ],
    loc="upper left",             # Anchor legend’s top-left corner
    bbox_to_anchor=(1.05, 1.0),   # X=1.05 (right of plot), Y=1.0 (top aligned)
    borderaxespad=0.0,             # removes extra padding between axes and legend
    #ncol=len(categories),        # Puts all categories in a single row
    frameon=False                # Optional: remove border box around legend
)



def get_n_toptiles(
    slide,
    category: str,
    output_dir: Path,
    coords: Tensor,
    scores: Tensor,
    stride: int,
    n: int = 8,
    full_resolution: bool = False,
) -> None:
    DEFAULT_TIFF_MPP = 0.25
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Slide MPP successfully retrieved from metadata: {slide_mpp}")
    except KeyError:
        slide_mpp = DEFAULT_TIFF_MPP
    

    (output_dir / f"toptiles_{category}").mkdir(exist_ok=True, parents=True)

    # determine the scaling factor between heatmap and original slide
    # 256 microns edge length by default, with 224px = ~1.14 MPP (± 10x magnification)
    feature_downsample_mpp = (
        256 / stride
    )  # NOTE: stride here only makes sense if the tiles were NON-OVERLAPPING
    scaling_factor = feature_downsample_mpp / slide_mpp

    top_score = scores.topk(n)

    # OPTIONAL: if the score is not larger than 0.5, it's indecisive on directionality
    # then add [top_score.values > 0.5]
    top_coords_downscaled = coords[top_score.indices]
    top_coords_original = np.uint(top_coords_downscaled * scaling_factor)

    # NOTE: target size (stride, stride) only works for NON-OVERLAPPING tiles
    # that were extracted in previous steps.
    for score_idx, pos in enumerate(top_coords_original):
        # Calculate the actual tile size in original slide coordinates
        tile_size_original = (np.uint(stride * scaling_factor), np.uint(stride * scaling_factor))
        
        tile = slide.read_region(
            (pos[0], pos[1]),
            0,
            tile_size_original,
        ).convert("RGB")
        
        # Only resize if not saving at full resolution
        if not full_resolution:
            tile = tile.resize((stride, stride))
        
        tile.save(
            (output_dir / f"toptiles_{category}")
            / f"score_{top_score.values[score_idx]:.2f}_toptiles_{category}_{(pos[0], pos[1])}.jpg"
        )


def main(
    slide_name: Sequence[str],
    feature_dir: Path,
    wsi_dir: Path,
    model_path: Path,
    output_dir: Path,
    n_toptiles: int = 8,
    overview: bool = True,
    full_resolution: bool = True,
    only_predicted_subtype: bool = True,  # Add this parameter
) -> None:
    learn = load_learner(model_path)
    learn.model.eval()
    categories: Collection[str] = learn.dls.train.dataset._datasets[
        -1
    ].encode.categories_[0]

    # for h5_path in feature_dir.glob(f"**/{slide_name}.h5"):
    print(f"slide_name: {slide_name}")
    print(type(slide_name))  # Should print <class 'list'>
    print(slide_name)  # Should print the full list of slide names

    for slide_iter in slide_name:
        print(f"Processing slide: {slide_iter}")
        for slide_path in wsi_dir.glob(f"**/{slide_iter}.*"):
            h5_path = feature_dir / slide_path.with_suffix(".h5").name
            slide_output_dir = output_dir / h5_path.stem
            slide_output_dir.mkdir(exist_ok=True, parents=True)
            print(f"Creating heatmaps for {slide_path.name}...")
            with h5py.File(h5_path) as h5:
                feats = torch.tensor(h5["feats"][:]).float()
                coords = torch.tensor(h5["coords"][:], dtype=torch.int)

            print(f"coords shape: {coords.shape}")
            print(f"feats shape: {feats.shape}")

            # stride is 224 using normal operations
            stride = get_stride(coords)

            print(f"stride: {stride}")

            preds, gradcam = gradcam_per_category(
                learn=learn, feats=feats, categories=categories
            )

            print(f"gradcam shape before permute: {gradcam.shape}")
            
            gradcam_2d = vals_to_im(gradcam.T, torch.div(coords, stride, rounding_mode='floor')).detach()
            #gradcam_2d = vals_to_im(gradcam.permute(-1, -2), torch.div(coords, stride, rounding_mode='floor')).detach()
            #gradcam_2d = vals_to_im(gradcam.mean(dim=2), torch.div(coords, stride, rounding_mode='floor')).detach()

            # Add this after calculating gradcam_2d
            plt.figure(figsize=(10, 10))
            plt.imshow(gradcam_2d.sum(-1).detach().cpu().numpy())
            plt.colorbar()
            plt.savefig(slide_output_dir / "raw_gradcam.png")
            plt.close()

            # scores = torch.softmax(
            #     learn.model(feats.unsqueeze(-2), torch.ones((len(feats)))), dim=1
            # )

            if feats.dim() == 3:
                # For Virchow2 features
                class_token = feats[:, 0]
                patch_tokens = feats[:, 5:]
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                scores = torch.softmax(
                    learn.model(embedding.unsqueeze(-2), torch.ones((len(feats)))), dim=1
                )
            else:
                # For other feature extractors
                scores = torch.softmax(
                    learn.model(feats.unsqueeze(-2), torch.ones((len(feats)))), dim=1
                )

            scores_2d = vals_to_im(scores, torch.div(coords, stride, rounding_mode='floor')).detach()
            
            '''            
            fig, axs = plt.subplots(nrows=2, ncols=max(2, len(categories)), figsize=(16, 10)) # 12, 8

            show_class_map(
                class_ax=axs[0, 1],
                top_scores=scores_2d.topk(2),
                gradcam_2d=gradcam_2d,
                categories=categories,
            )

            slide = load_slide_ext(slide_path)

            for ax, (pos_idx, category) in zip(axs[1, :], enumerate(categories)):
                ax: Axes
                topk = scores_2d.topk(2)
                category_support = torch.where(
                    # To get the "positiveness",
                    # it checks whether the "hot" class has the highest score for each pixel
                    topk.indices[..., 0] == pos_idx,
                    # Then, if the hot class has the highest score,
                    # it assigns a positive value based on its difference from the second highest score
                    scores_2d[..., pos_idx] - topk.values[..., 1],
                    # Likewise, if it has NOT a negative value based on the difference of that class' score to the highest one
                    scores_2d[..., pos_idx] - topk.values[..., 0],
                )

                # So, if we have a pixel with scores (.4, .4, .2) and would want to get the heat value for the first class,
                # we would get a neutral color, because it is matched with the second class
                # But if our scores were (.4, .3, .3), it would be red,
                # because now our class is .1 above its nearest competitor

                attention = torch.where(
                    topk.indices[..., 0] == pos_idx,
                    gradcam_2d[..., pos_idx] / gradcam_2d.max(),
                    (
                        others := gradcam_2d[
                            ..., list(set(range(len(categories))) - {pos_idx})
                        ]
                        .max(-1)
                        .values
                    )
                    / others.max(),
                )

                score_im = plt.get_cmap("RdBu")(
                    -category_support * attention / attention.max() / 2 + 0.5
                )

                score_im[..., -1] = attention > 0

                ax.imshow(score_im)
                ax.set_title(f"{category} {preds[0,pos_idx]:1.2f}")
                target_size=np.array(score_im.shape[:2][::-1]) * 8
                # latest PIL requires shape to be a tuple (), not array []
                Image.fromarray(np.uint8(score_im * 255)).resize(
                    tuple(target_size), resample=Image.NEAREST
                ).save(
                    slide_output_dir
                    / f"scores-{h5_path.stem}--score_{category}={preds[0][pos_idx]:0.2f}.png"
                )

                get_n_toptiles(
                    slide=slide,
                    category=category,
                    stride=stride,
                    output_dir=slide_output_dir,
                    scores=scores[:, pos_idx],
                    coords=coords,
                    n=n_toptiles,
                    full_resolution=full_resolution,
                )

            print(f"overview: {overview}")
            if overview:
                thumb = show_thumb(
                    slide=slide,
                    thumb_ax=axs[0, 0],
                    attention=attention,
                )
                Image.fromarray(thumb).save(slide_output_dir / f"thumbnail-{h5_path.stem}.png")

                for ax in axs.ravel():
                    ax.axis("off")

                fig.savefig(slide_output_dir / f"overview-{h5_path.stem}.png", dpi=300)
                plt.close(fig)
            '''

            # ============================
            # PLOT 1: Thumbnail + Class Map + Legend
            # ============================
            fig1, axs1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            
            slide = load_slide_ext(slide_path)

            thumb = show_thumb(
                slide=slide,
                thumb_ax=axs1[0],
                attention=gradcam_2d,
            )
            axs1[0].set_title("Slide Thumbnail")
            axs1[0].axis("off")

            show_class_map(
                class_ax=axs1[1],
                top_scores=scores_2d.topk(2),
                gradcam_2d=gradcam_2d,
                categories=categories,
            )
            axs1[1].set_title("Class Prediction Map")
            axs1[1].axis("off")

            fig1.tight_layout()
            fig1.savefig(slide_output_dir / f"overview_thumbnail_classmap-{h5_path.stem}.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)

            # ============================
            # PLOT 2: One heatmap per class with class-specific legends
            # ============================
            n_classes = len(categories)
            fig2, axs2 = plt.subplots(nrows=1, ncols=n_classes, figsize=(5 * n_classes, 5), constrained_layout=True)

            if n_classes == 1:
                axs2 = [axs2]  # wrap in list for consistent iteration

            # Determine the predicted subtype (class with highest prediction)
            predicted_class_idx = preds[0].argmax().item()
            predicted_category = categories[predicted_class_idx]
            print(f"Predicted subtype: {predicted_category} (confidence: {preds[0, predicted_class_idx]:.3f})")

            for ax, (pos_idx, category) in zip(axs2, enumerate(categories)):
                ax: Axes
                topk = scores_2d.topk(2)
                category_support = torch.where(
                    topk.indices[..., 0] == pos_idx,
                    scores_2d[..., pos_idx] - topk.values[..., 1],
                    scores_2d[..., pos_idx] - topk.values[..., 0],
                )

                attention = torch.where(
                    topk.indices[..., 0] == pos_idx,
                    gradcam_2d[..., pos_idx] / gradcam_2d.max(),
                    (
                        others := gradcam_2d[
                            ..., list(set(range(len(categories))) - {pos_idx})
                        ]
                        .max(-1)
                        .values
                    ) / others.max(),
                )

                score_im = plt.get_cmap("RdBu")(
                    -category_support * attention / attention.max() / 2 + 0.5
                )
                score_im[..., -1] = attention > 0

                ax.imshow(score_im)
                ax.set_title(f"{category} ({preds[0, pos_idx]:.2f})")
                ax.axis("off")

                # Save the heatmap
                target_size=np.array(score_im.shape[:2][::-1]) * 8
                Image.fromarray(np.uint8(score_im * 255)).resize(
                    tuple(target_size), resample=Image.NEAREST
                ).save(
                    slide_output_dir
                    / f"scores-{h5_path.stem}--score_{category}={preds[0][pos_idx]:0.2f}.png"
                )

                # Save top tiles only for predicted subtype if requested
                if only_predicted_subtype:
                    if pos_idx == predicted_class_idx:
                        print(f"Saving top tiles for predicted subtype: {category}")
                        get_n_toptiles(
                            slide=slide,
                            category=category,
                            stride=stride,
                            output_dir=slide_output_dir,
                            scores=scores[:, pos_idx],
                            coords=coords,
                            n=n_toptiles,
                            full_resolution=full_resolution,
                        )
                else:
                    # Save top tiles for all categories (original behavior)
                    get_n_toptiles(
                        slide=slide,
                        category=category,
                        stride=stride,
                        output_dir=slide_output_dir,
                        scores=scores[:, pos_idx],
                        coords=coords,
                        n=n_toptiles,
                        full_resolution=full_resolution,
                    )

            # Create a custom colormap legend explaining the RdBu scale
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            # Normalize the range from -1 to 1, matching your score scaling logic
            norm = Normalize(vmin=-1, vmax=1)
            sm = ScalarMappable(cmap="RdBu", norm=norm)
            sm.set_array([])

            # Add colorbar to the figure — place it to the right
            cbar = fig2.colorbar(
                sm,
                ax=axs2,
                orientation="horizontal",
                fraction=0.05,
                pad=0.01
            )
            cbar.set_label("Support for Class (Red = strong, Blue = against)")

            #fig2.tight_layout()
            fig2.savefig(slide_output_dir / f"overview_classheatmaps-{h5_path.stem}.png", dpi=300, bbox_inches="tight")
            plt.close(fig2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser("heatmaps")
    parser.add_argument(
        "--slide-name",
        metavar="PATH",
        type=str,
        required=True,
        help="Name of the WSI to create heatmap for (no extensions)",
    )
    parser.add_argument(
        "--wsi-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the SVSs",
    )
    parser.add_argument(
        "--feature-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the slides' features",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to the trained model's export.pkl",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory to save the heatmaps to",
    )
    parser.add_argument(
        "--n-toptiles",
        type=int,
        default=8,
        required=False,
        help="Number of toptiles to generate, 8 by default",
    )
    parser.add_argument(
        "--overview",
        type=bool,
        default=True,
        required=False,
        help="Generate final overview image",
    )
    parser.add_argument(
        "--full-resolution",
        action="store_true",
        help="Save top tiles at full/native resolution instead of resizing to stride size",
    )
    parser.add_argument(
        "--only-predicted-subtype",
        action="store_true",
        help="Save top tiles only for the predicted subtype instead of all categories",
    )
    args = parser.parse_args()
    main(**vars(args))
