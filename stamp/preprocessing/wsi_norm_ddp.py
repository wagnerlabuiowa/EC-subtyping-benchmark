import os
import argparse
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import openslide
import PIL
import cv2
import numpy as np
import time
from datetime import timedelta
from random import shuffle
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from .helpers import stainNorm_Macenko
from .helpers.common import supported_extensions
from .helpers.concurrent_canny_rejection import reject_background
from .helpers.loading_slides import process_slide_jpg, load_slide, get_raw_tile_list
from .helpers.exceptions import MPPExtractionError
from .helpers.feature_extractors import (
    FeatureExtractorCTP, FeatureExtractorUNI, FeatureExtractorUNI2, FeatureExtractorVirchow2,
    FeatureExtractorHOptimus0, FeatureExtractorProvGigaPath, FeatureExtractorViTB16, extract_features_
)
import h5py

PIL.Image.MAX_IMAGE_PIXELS = None

def clean_lockfile(file):
    if os.path.exists(file): # Catch collision cases
        os.remove(file)

@contextmanager
def lock_file(slide_path: Path):
    try:
        Path(f"{slide_path}.lock").touch()
    except PermissionError:
        pass # No write permissions for wsi directory
    try:
        yield
    finally:
        clean_lockfile(f"{slide_path}.lock")

def test_wsidir_write_permissions(wsi_dir: Path):
    try:
        testfile = wsi_dir/f"test_{str(os.getpid())}.tmp"
        Path(testfile).touch()
    except PermissionError:
        logging.warning("No write permissions for wsi directory! If multiple stamp processes are running "
                        "in parallel, the final summary may show an incorrect number of slides processed.")
    finally:
        clean_lockfile(testfile)

def save_image(image, path: Path):
    width, height = image.size
    if width > 65500 or height > 65500:
        logging.warning(f"Image size ({width}x{height}) exceeds maximum size of 65500x65500, "
                        f"{path.name} will not be cached...")
        return
    image.save(path)

# Function to initialize DDP
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Function to clean up DDP resources
def cleanup_ddp():
    dist.destroy_process_group()

# Function to run DDP-based preprocessing
def run_ddp_feature_extraction(cfg, model_path):
    if (cfg.preprocessing.microns == 112):
        print("Downscaling to 2 processes(world_size) with 112 micron preprocessing")
        world_size = 2 ### testing for 112 micron runs 
    else:
        world_size = torch.cuda.device_count()  # Number of GPUs

    # Load normalization template once before spawning processes
    normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
    if cfg.preprocessing.norm:
        print("\nLoading Macenko normalisation template...")
        target = cv2.imread(str(normalization_template_path))
        if target is None:
            raise ValueError(f"Failed to load normalization template from {normalization_template_path}")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    else:
        target = None

    # Get list of slides, filter out slides that have already been processed
    img_dir = [svs for ext in supported_extensions for svs in Path(cfg.preprocessing.wsi_dir).glob(f"**/*{ext}")]

    shuffle(img_dir)
    total_slides = len(img_dir)
    
    # Step 1: Initialize the list of GPUs
    slide_assignments = [[] for _ in range(world_size)]

    # Step 2: Distribute slides in a round-robin fashion
    for idx, slide in enumerate(img_dir):
        gpu_idx = idx % world_size  # This will cycle through GPUs
        slide_assignments[gpu_idx].append(slide)

    print(f"Distributed {total_slides} slides for preprocessing on {world_size} GPUs.")

    # Pass the loaded template to each process
    mp.spawn(preprocess_ddp, args=(world_size, cfg, model_path, slide_assignments, target), nprocs=world_size, join=True)

# Preprocess function with DDP integrated
def preprocess_ddp(rank, world_size, cfg, model_path, slide_assignments, target_image=None):
    # Setup DDP for each process
    setup_ddp(rank, world_size)

    # Extract parameters from config
    output_dir = Path(cfg.preprocessing.output_dir)
    wsi_dir = Path(cfg.preprocessing.wsi_dir)
    model_path = Path(model_path)
    cache_dir = Path(cfg.preprocessing.cache_dir)
    norm = cfg.preprocessing.norm
    del_slide = cfg.preprocessing.del_slide
    only_feature_extraction = cfg.preprocessing.only_feature_extraction
    keep_dir_structure=cfg.preprocessing.keep_dir_structure if 'keep_dir_structure' in cfg.preprocessing else False
    cache = cfg.preprocessing.cache
    feat_extractor = cfg.preprocessing.feat_extractor
    log_only_filtering = False # Set to True to only log filtering statistics without extracting features

    # Clean up potentially old leftover .lock files
    for lockfile in wsi_dir.glob("**/*.lock"):
        if time.time() - os.path.getmtime(lockfile) > 20:
            clean_lockfile(lockfile)
    has_gpu = torch.cuda.is_available()
    target_microns = cfg.preprocessing.microns
    patch_size = int(224)
    target_mpp = target_microns/patch_size
    patch_shape = (patch_size, patch_size)
    step_size = patch_size
    cores = cfg.preprocessing.cores
    device = f'cuda:{rank}'

    if not log_only_filtering:
    # Initialize the feature extraction model
        if feat_extractor == "ctp":
            extractor = FeatureExtractorCTP(checkpoint_path=model_path)
        elif feat_extractor == "uni":
            extractor = FeatureExtractorUNI()
        elif feat_extractor == "uni2":
            extractor = FeatureExtractorUNI2()
        elif feat_extractor == "virchow2":
            extractor = FeatureExtractorVirchow2()
        elif feat_extractor == "h-optimus-0":
            extractor = FeatureExtractorHOptimus0()
        elif feat_extractor == "prov-gigapath":
            extractor = FeatureExtractorProvGigaPath()
        elif feat_extractor == "ViT-B16":
            extractor = FeatureExtractorViTB16()
        else:
            raise Exception(f"Invalid feature extractor '{feat_extractor}' selected")
        model_name = extractor.init_feat_extractor(device=device)
    else:
        model_name = "log_only_mode"
        logging.info("LOG-ONLY MODE: Will log filtering statistics without extracting features")
    



    # Create cache and output directories
    if cache:
        cache_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_method = "STAMP_macenko_" if norm else "STAMP_raw_"
    model_name_norm = Path(norm_method + model_name)
    output_file_dir = output_dir/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)

    # Create logfile and set up logging
    logfile_name = "logfile_" + time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(os.getpid())
    logdir = output_file_dir/logfile_name
    logging.basicConfig(filename=logdir, force=True, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Preprocessing started at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(f"Norm: {norm} | Target_microns: {target_microns} | Patch_size: {patch_size} | MPP: {target_mpp}")
    logging.info(f"Model: {model_name}\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Stored logfile in {logdir}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")

    # Tracking variables for filtering statistics
    total_slides_processed = 0
    slides_with_canny_rejection = []
    slides_filtered_by_cached_black_patches = []

    if norm:
        print(f"\nInitialising Macenko normaliser on GPU {rank}...")
        normalizer = stainNorm_Macenko.Normalizer()
        # Use the pre-loaded target image instead of loading it again
        if target_image is not None:
            normalizer.fit(target_image)
        else:
            # Fallback if target_image wasn't passed correctly
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            print(f"Loading template from {normalization_template_path} on GPU {rank}")
            target = cv2.imread(str(normalization_template_path))
            if target is None:
                raise ValueError(f"Failed to load normalization template from {normalization_template_path}")
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            normalizer.fit(target)

    test_wsidir_write_permissions(wsi_dir)

    total_start_time = time.time()

    img_name = "norm_slide.jpg" if norm else "canny_slide.jpg"
    # Get list of slides, filter out slides that have already been processed
    print("Scanning for existing feature files...")
    existing = [f.stem for f in output_file_dir.glob("**/*.h5")] if output_file_dir.exists() else []

    rank_img_dir = slide_assignments[rank]
    if not only_feature_extraction:
        existing = [f for f in existing if f in [f.stem for f in rank_img_dir]]
        rank_img_dir = [f for f in rank_img_dir if f.stem not in existing]
    else:
        if not cache_dir.exists():
            logging.error("Cache directory does not exist, cannot extract features from cached slides!")
            exit(1)
        rank_img_dir = [jpg for jpg in cache_dir.glob(f"**/*/{img_name}")]
        existing = [f for f in existing if f in [f.parent.name for f in rank_img_dir]]
        rank_img_dir = [f for f in rank_img_dir if f.parent.name not in existing]

    num_total = len(rank_img_dir) + len(existing)
    num_processed = 0
    error_slides = []
    if len(existing):
        print(f"\n For {len(existing)} out of {num_total} slides in the wsi directory feature files were found, skipping these slides...")

    for slide_url in tqdm(rank_img_dir, desc=f"Preprocessing on GPU {rank}", leave=False, miniters=1, mininterval=0):
        slide_name = slide_url.stem if not only_feature_extraction else slide_url.parent.name
        slide_cache_dir = cache_dir / slide_name
        if cache:
            slide_cache_dir.mkdir(parents=True, exist_ok=True)
    
        print("\n")
        logging.info(f"===== Processing slide {slide_name} on GPU {rank} =====")
        slide_subdir = slide_url.parent.relative_to(wsi_dir)
        if not keep_dir_structure or slide_subdir == Path("."):
            feat_out_dir = output_file_dir/slide_name
        else:
            (output_file_dir/slide_subdir).mkdir(parents=True, exist_ok=True)
            feat_out_dir = output_file_dir/slide_subdir/slide_name
        if not (os.path.exists((f"{feat_out_dir}.h5"))) and not os.path.exists(f"{slide_url}.lock"):
            with lock_file(slide_url):
                canny_rejected_count = 0
                total_tiles_before_filtering = 0
                total_tiles_after_filtering = 0
                cached_black_patches_filtered = 0
                
                if (
                    (only_feature_extraction and (slide_jpg := slide_url).exists()) or \
                    (slide_jpg := slide_cache_dir/"norm_slide.jpg").exists()
                ):
                    canny_norm_patch_list, coords_list, total = process_slide_jpg(slide_jpg)
                    total_tiles_before_filtering = total
                    total_tiles_after_filtering = len(canny_norm_patch_list)
                    cached_black_patches_filtered = total - len(canny_norm_patch_list)

                    print(f"Loaded {img_name}, {len(canny_norm_patch_list)}/{total} tiles remain")
                    logging.info(f"CACHED SLIDE FILTERING: {total_tiles_before_filtering} total tiles, "
                                f"{cached_black_patches_filtered} black patches filtered, "
                                f"{total_tiles_after_filtering} tiles remaining")

                    if cached_black_patches_filtered > 0:
                        slides_filtered_by_cached_black_patches.append({
                            'slide': slide_name,
                            'total_tiles': total_tiles_before_filtering,
                            'filtered': cached_black_patches_filtered,
                            'remaining': total_tiles_after_filtering
                        })
                else:
                    try:
                        slide = openslide.OpenSlide(slide_url)
                    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                        logging.error("Unsupported format for slide, continuing...")
                        error_slides.append(slide_name)
                        continue
                    except Exception as e:
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue

                    start_time = time.time()
                    try:
                        slide_array = load_slide(slide=slide, target_mpp=target_mpp, cores=cores)
                    except MPPExtractionError:
                        if del_slide:
                            logging.error("MPP missing in slide metadata, deleting slide and continuing...")
                            if os.path.exists(slide_url):
                                os.remove(slide_url)
                        else:
                            logging.error("MPP missing in slide metadata, continuing...")
                        error_slides.append(slide_name)
                        continue
                    except openslide.lowlevel.OpenSlideError as e:
                        print("")
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue

                    # Remove .SVS from memory
                    del slide                    
                    print(f"\nLoaded slide: {time.time() - start_time:.2f} seconds")
                    print(f"\nSize of WSI: {slide_array.shape}")
                        
                    if cache:
                        # Save raw .svs jpg
                        raw_image = PIL.Image.fromarray(slide_array)
                        save_image(raw_image, slide_cache_dir/"slide.jpg")

                    #Do edge detection here and reject unnecessary tiles BEFORE normalisation
                    bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img=slide_array, patch_size=patch_shape, step=step_size, cores=cores)

                    # Track Canny rejection statistics
                    canny_rejected_count = int(np.sum(rejected_tile_array))
                    total_tiles_before_filtering = len(rejected_tile_array)
                    total_tiles_after_filtering = total_tiles_before_filtering - canny_rejected_count
                    
                    logging.info(f"CANNY REJECTION FILTERING: {total_tiles_before_filtering} total tiles, "
                                f"{canny_rejected_count} tiles rejected by Canny edge detection, "
                                f"{total_tiles_after_filtering} tiles remaining")
                    
                    slides_with_canny_rejection.append({
                        'slide': slide_name,
                        'total_tiles': total_tiles_before_filtering,
                        'rejected_by_canny': canny_rejected_count,
                        'remaining': total_tiles_after_filtering,
                        'rejection_rate': canny_rejected_count / total_tiles_before_filtering if total_tiles_before_filtering > 0 else 0
                    })
                    start_time = time.time()
                    # Pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation
                    if norm:
                        print(f"Normalising slide...")
                        canny_img, img_norm_wsi_jpg, canny_norm_patch_list, coords_list = normalizer.transform(slide_array, bg_reject_array, 
                                                                                                            rejected_tile_array, patch_shapes, target_mpp, cores=cores)
                        print(f"\nNormalised slide: {time.time() - start_time:.2f} seconds")
                        if cache:
                            save_image(img_norm_wsi_jpg, slide_cache_dir/"norm_slide.jpg")
                    else:
                        canny_img, canny_norm_patch_list, coords_list = get_raw_tile_list(slide_array.shape, bg_reject_array,
                                                                                        rejected_tile_array, patch_shapes, target_mpp)

                    if cache:
                        print("Saving Canny background rejected image...")
                        save_image(canny_img, slide_cache_dir/"canny_slide.jpg")

                    # Remove original slide jpg from memory
                    del slide_array
                    
                    # Optionally remove the original slide from harddrive
                    if del_slide:
                        print("Deleting slide from local folder...")
                        if os.path.exists(slide_url):
                            os.remove(slide_url)

                # Skip feature extraction if in log-only mode
                if log_only_filtering:
                    logging.info(f"LOG-ONLY MODE: Skipping feature extraction for {slide_name}")
                    logging.info(f"Filtering summary for {slide_name}: "
                                f"{total_tiles_after_filtering} tiles remaining after filtering")
                    num_processed += 1
                    total_slides_processed += 1
                else:
                    # Original feature extraction code
                    print(f"\nExtracting {model_name} features from slide on GPU {rank}...")
                    start_time = time.time()
                    if len(canny_norm_patch_list) > 0:
                        extract_features_(model=extractor.model, transform=extractor.transform, model_name=model_name,
                                        norm_wsi_img=canny_norm_patch_list, coords=coords_list, wsi_name=slide_name,
                                        outdir=feat_out_dir, cores=cores, is_norm=norm, device=device if has_gpu else "cpu",
                                        target_microns=target_microns, patch_size=patch_size)
                        logging.info(f"Extracted features from slide on GPU {rank}: {time.time() - start_time:.2f} seconds ({len(canny_norm_patch_list)} tiles)")
                        num_processed += 1

                        # with h5py.File(f'{feat_out_dir}.h5', 'w') as f:
                        #     f['coords'] = coords_list       # in µm or lvl-0 px
                        #     f['feats']  = canny_norm_patch_list
                        #     f.attrs['mpp']         = target_mpp      #  µm / pixel of THIS run
                        #     f.attrs['patch_size']  = patch_size      #  = 224
                        # continue
                    else:
                        logging.error("0 tiles remain to extract features from after pre-processing. Continuing...")
                        error_slides.append(slide_name)
                        continue

                
        else:
            if os.path.exists((f"{feat_out_dir}.h5")):
                logging.info(".h5 file for this slide already exists. Skipping...")
            else:
                logging.info("Slide is already being processed. Skipping...")
            existing.append(slide_name)
            if del_slide:
                print("Deleting slide from local folder...")
                if os.path.exists(slide_url):
                    os.remove(slide_url)

    cleanup_ddp()

    # Log summary statistics
    logging.info(f"===== End-to-end processing time of {num_total} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} =====")
    
    if log_only_filtering:
        logging.info("=" * 80)
        logging.info("FILTERING STATISTICS SUMMARY (LOG-ONLY MODE)")
        logging.info("=" * 80)
        logging.info(f"Total slides processed: {total_slides_processed}")
        logging.info(f"Slides with Canny rejection applied: {len(slides_with_canny_rejection)}")
        
        if slides_with_canny_rejection:
            total_canny_rejected = sum(s['rejected_by_canny'] for s in slides_with_canny_rejection)
            total_canny_tiles = sum(s['total_tiles'] for s in slides_with_canny_rejection)
            avg_rejection_rate = sum(s['rejection_rate'] for s in slides_with_canny_rejection) / len(slides_with_canny_rejection)
            
            logging.info(f"Total tiles processed with Canny rejection: {total_canny_tiles}")
            logging.info(f"Total tiles rejected by Canny: {total_canny_rejected}")
            logging.info(f"Average Canny rejection rate: {avg_rejection_rate:.2%}")
            logging.info("\nPer-slide Canny rejection details:")
            for slide_info in slides_with_canny_rejection:
                logging.info(f"  {slide_info['slide']}: {slide_info['rejected_by_canny']}/{slide_info['total_tiles']} "
                           f"tiles rejected ({slide_info['rejection_rate']:.2%})")
        
        logging.info(f"Slides filtered by cached black patches: {len(slides_filtered_by_cached_black_patches)}")
        if slides_filtered_by_cached_black_patches:
            total_cached_filtered = sum(s['filtered'] for s in slides_filtered_by_cached_black_patches)
            total_cached_tiles = sum(s['total_tiles'] for s in slides_filtered_by_cached_black_patches)
            logging.info(f"Total tiles from cached slides: {total_cached_tiles}")
            logging.info(f"Total black patches filtered from cached slides: {total_cached_filtered}")
            logging.info("\nPer-slide cached filtering details:")
            for slide_info in slides_filtered_by_cached_black_patches:
                logging.info(f"  {slide_info['slide']}: {slide_info['filtered']}/{slide_info['total_tiles']} "
                           f"black patches filtered")
        logging.info("=" * 80)
    
    logging.info(f"Summary: Processed {num_processed} slides, encountered {len(error_slides)} errors, skipped {len(existing)} readily-processed slides")
    if len(error_slides):
        logging.info("The following slides were not processed due to errors:\n\n" + "\n".join(error_slides))
