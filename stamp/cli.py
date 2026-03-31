from omegaconf import OmegaConf, DictConfig
from omegaconf.listconfig import ListConfig
import argparse
from pathlib import Path
import os
from typing import Iterable, Optional
import shutil
import copy

import torch
from torchvision import transforms
import timm
from huggingface_hub import login
from timm.layers import SwiGLUPacked

NORMALIZATION_TEMPLATE_URL = "https://github.com/Avic3nna/STAMP/blob/main/resources/normalization_template.jpg?raw=true"
CTRANSPATH_WEIGHTS_URL = "https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download"
DEFAULT_RESOURCES_DIR = Path(__file__).with_name("resources")
DEFAULT_CONFIG_FILE = Path("config.yaml")
STAMP_FACTORY_SETTINGS = Path(__file__).with_name("config.yaml")

class ConfigurationError(Exception):
    pass

def check_path_exists(path):
    directories = path.split(os.path.sep)
    current_path = os.path.sep
    for directory in directories:
        current_path = os.path.join(current_path, directory)
        if not os.path.exists(current_path):
            return False, directory
    return True, None


def check_and_handle_path(path, path_key, prefix):
    exists, directory = check_path_exists(path)
    if not exists:
        print(f"From input path: '{path}'")
        print(f"Directory '{directory}' does not exist.")
        print(f"Check the input path of '{path_key}' from the '{prefix}' section.")
        raise SystemExit(f"Stopping {prefix} due to faulty user input...")


def _config_has_key(cfg: DictConfig, key: str):
    try:
        for k in key.split("."):
            cfg = cfg[k]
        if cfg is None:
            return False
    except KeyError:
        return False
    return True

def require_configs(cfg: DictConfig, keys: Iterable[str], prefix: Optional[str] = None,
                    paths_to_check: Iterable[str] = []):
    keys = [f"{prefix}.{k}" for k in keys]
    missing = [k for k in keys if not _config_has_key(cfg, k)]
    if len(missing) > 0:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")

    # Check if paths exist
    for path_key in paths_to_check:
        try:
            #for all but modeling.statistics
            path = cfg[prefix][path_key]
        except:
            #for modeling.statistics, handling the pred_csvs
            path = OmegaConf.select(cfg, f"{prefix}.{path_key}")
        if isinstance(path, ListConfig):
            for p in path:
                check_and_handle_path(p, path_key, prefix)
        else:
            check_and_handle_path(path, path_key, prefix)


def create_config_file(config_file: Optional[Path]):
    """Create a new config file at the specified path (by copying the default config file)."""
    config_file = config_file or DEFAULT_CONFIG_FILE
    # Locate original config file
    if not STAMP_FACTORY_SETTINGS.exists():
        raise ConfigurationError(f"Default STAMP config file not found at {STAMP_FACTORY_SETTINGS}")
    # Copy original config file
    shutil.copy(STAMP_FACTORY_SETTINGS, config_file)
    print(f"Created new config file at {config_file.absolute()}")

def resolve_config_file_path(config_file: Optional[Path]) -> Path:
    """Resolve the path to the config file, falling back to the default config file if not specified."""
    if config_file is None:
        if DEFAULT_CONFIG_FILE.exists():
            config_file = DEFAULT_CONFIG_FILE
        else:
            config_file = STAMP_FACTORY_SETTINGS
            print(f"Falling back to default STAMP config file because {DEFAULT_CONFIG_FILE.absolute()} does not exist")
            if not config_file.exists():
                raise ConfigurationError(f"Default STAMP config file not found at {config_file}")
    if not config_file.exists():
        raise ConfigurationError(f"Config file {Path(config_file).absolute()} not found (run `stamp init` to create the config file or use the `--config` flag to specify a different config file)")
    return config_file

def huggingface_login():
    """Login to Hugging Face using token-based authentication."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set. "
            "Get a read-only token from https://huggingface.co/settings/tokens "
            "and run: export HF_TOKEN=hf_..."
        )
    login(token=token)

def run_cli(args: argparse.Namespace):
    # Handle init command
    if args.command == "init":
        create_config_file(args.config)
        return

    # Load YAML configuration
    config_file = resolve_config_file_path(args.config)
    cfg = OmegaConf.load(config_file)

    # Set environment variables
    if "STAMP_RESOURCES_DIR" not in os.environ:
        os.environ["STAMP_RESOURCES_DIR"] = str(DEFAULT_RESOURCES_DIR)
    
    match args.command:
        case "init":
            return # this is handled above
        case "setup":
            # login to huggingface
            huggingface_login()

            # Download normalization template
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            normalization_template_path.parent.mkdir(parents=True, exist_ok=True)
            if normalization_template_path.exists():
                print(f"Skipping download, normalization template already exists at {normalization_template_path}")
            else:
                print(f"Downloading normalization template to {normalization_template_path}")
                import requests
                r = requests.get(NORMALIZATION_TEMPLATE_URL)
                with normalization_template_path.open("wb") as f:
                    f.write(r.content)

            # Download feature extractor model
            feat_extractor = cfg.preprocessing.feat_extractor
            if feat_extractor == 'ctp':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/ctranspath.pth")
            elif feat_extractor == 'uni':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/uni/pytorch_model.bin")
            elif feat_extractor == 'uni2':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/uni2/pytorch_model.bin")
            elif feat_extractor == 'virchow2':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/virchow2/pytorch_model.bin")  
            elif feat_extractor == 'h-optimus-0':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/h-optimus-0/pytorch_model.bin")  
            elif feat_extractor == 'prov-gigapath':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/prov-gigapath/pytorch_model.bin")
            elif feat_extractor == 'ViT-B16':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/ViT-B16/pytorch_model.bin")
            else:
                raise ConfigurationError(f"Unknown feature extractor model: {feat_extractor}")
            
            
            model_path.parent.mkdir(parents=True, exist_ok=True)
            device = cfg.preprocessing.device

            # Download feature extractor model
            if model_path.exists():
                print(f"Skipping download, feature extractor model already exists at {model_path}")
            else:
                if feat_extractor == 'ctp':
                    print(f"Downloading CTransPath weights (gdown)")
                    import gdown
                    gdown.download(CTRANSPATH_WEIGHTS_URL, str(model_path))              
                elif feat_extractor == 'uni':
                    print(f"Loading UNI weights (timm)")
                    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
                    torch.save(model, model_path)
                elif feat_extractor == 'uni2':
                    print(f"Loading UNI2 weights (timm)")
                    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
                    timm_kwargs = {
                                'img_size': 224, 
                                'patch_size': 14, 
                                'depth': 24,
                                'num_heads': 24,
                                'init_values': 1e-5, 
                                'embed_dim': 1536,
                                'mlp_ratio': 2.66667*2,
                                'num_classes': 0, 
                                'no_embed_class': True,
                                'mlp_layer': timm.layers.SwiGLUPacked, 
                                'act_layer': torch.nn.SiLU, 
                                'reg_tokens': 8, 
                                'dynamic_img_size': True
                            }
                    model = timm.create_model("hf-hub:MahmoodLab/uni2-h", pretrained=True, **timm_kwargs)
                    torch.save(model, model_path)
                elif feat_extractor == 'virchow2':
                    print(f"Loading Virchow2 weights (timm)")
                    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
                    torch.save(model, model_path)
                elif feat_extractor == 'h-optimus-0':
                    print(f"Loading H-Optimus-0 weights (timm)")
                    print(f"model_path: {model_path}")
                    print(f"device: {device}")
                    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
                    torch.save(model, model_path)
                elif feat_extractor == 'prov-gigapath':
                    print(f"Loading Prov-Gigapath weights (timm)")
                    print(f"model_path: {model_path}")
                    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
                    torch.save(model, model_path)
                elif feat_extractor == 'ViT-B16':
                    print(f"Loading ViT-B16 weights (timm)")
                    print(f"model_path: {model_path}")
                    model = timm.create_model("vit_large_patch16_224.orig_in21k", pretrained=True)
                    torch.save(model, model_path)
                else:
                    raise ConfigurationError(f"Unknown feature extractor model: {feat_extractor}")

        # print config  
        case "config":
            print(OmegaConf.to_yaml(cfg, resolve=True))

        # preprocess
        case "preprocess":
            require_configs(
                cfg,
                ["output_dir", "wsi_dir", "cache_dir", "microns", "cores", "norm", "del_slide", "only_feature_extraction", "device", "feat_extractor"],
                prefix="preprocessing",
                paths_to_check=["wsi_dir"]
            )
            c = cfg.preprocessing
            # Some checks
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            if c.norm and not Path(normalization_template_path).exists():
                raise ConfigurationError(f"Normalization template {normalization_template_path} does not exist, please run `stamp setup` to download it.")
            if c.feat_extractor == 'ctp':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/ctranspath.pth"
            elif c.feat_extractor == 'uni':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/uni/pytorch_model.bin"
            elif c.feat_extractor == 'uni2':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/uni2/pytorch_model.bin"
            elif c.feat_extractor == 'virchow2':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/virchow2/pytorch_model.bin"
            elif c.feat_extractor == 'h-optimus-0':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/h-optimus-0/pytorch_model.bin"
            elif c.feat_extractor == 'prov-gigapath':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/prov-gigapath/pytorch_model.bin"
            elif c.feat_extractor == 'ViT-B16':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/ViT-B16/pytorch_model.bin"
            else:
                raise ConfigurationError(f"Unknown feature extractor model: {c.feat_extractor}")
                
            if not Path(model_path).exists():
                raise ConfigurationError(f"Feature extractor model {model_path} does not exist, please run `stamp setup` to download it.")
            
            # preprocess whole slide images
            from .preprocessing.wsi_norm import preprocess
                        
            # Determine available GPUs
            if torch.cuda.is_available():
                device = c.device
                print(f"Using {device} for preprocessing")
                # num_gpus = torch.cuda.device_count()
                # if num_gpus > 1:
                #     print(f"Using {num_gpus} GPUs for preprocessing")
                #     device = [f"cuda:{i}" for i in range(num_gpus)]
                # else:
                #     print("Using 1 GPU for preprocessing")
                #     device = "cuda:0"
            else:
                print("No GPUs available, using CPU")
                device = "cpu"
            
            preprocess(
                output_dir=Path(c.output_dir),
                wsi_dir=Path(c.wsi_dir),
                model_path=Path(model_path),
                cache_dir=Path(c.cache_dir),
                feat_extractor=c.feat_extractor,
                target_microns=c.microns,
                cores=c.cores,
                norm=c.norm,
                del_slide=c.del_slide,
                cache=c.cache if 'cache' in c else True,
                only_feature_extraction=c.only_feature_extraction,
                keep_dir_structure=c.keep_dir_structure if 'keep_dir_structure' in c else False,
                device=device,
                normalization_template=normalization_template_path
            )
            
        case "train":
            require_configs(
                cfg,
                ["clini_table", "slide_table", "output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels", "loss_type"],
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "feature_dir"]
            )
            c = cfg.modeling
            
            # Check if c.checkpoint is provided; if not, set it to a default value or None
            if c.checkpoint is not None:
                checkpoint = Path(c.checkpoint)
            else:
                checkpoint = None  # or set to a default Path if needed

            device = getattr(c, "device", None)

            from .modeling.marugoto.transformer.helpers import train_categorical_model_
            train_categorical_model_(clini_table=Path(c.clini_table), 
                                     slide_table=Path(c.slide_table),
                                     feature_dir=Path(c.feature_dir), 
                                     output_path=Path(c.output_dir),
                                     target_label=c.target_label, 
                                     cat_labels=c.cat_labels,
                                     cont_labels=c.cont_labels, 
                                     categories=c.categories, 
                                     loss_type=c.loss_type,
                                     model_type=c.model_type,  
                                     use_coords=c.use_coords,
                                     bag_size=c.bag_size,
                                     checkpoint=checkpoint,
                                     device=device)
        case "crossval":
            require_configs(
                cfg,
                ["clini_table", "slide_table", "output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels", "n_splits", "loss_type", "model_type"], # this one requires the n_splits key!
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "feature_dir"]
            )
            c = cfg.modeling

            # Check if c.checkpoint is provided; if not, set it to a default value or None
            if c.checkpoint is not None:
                checkpoint = Path(c.checkpoint)
            else:
                checkpoint = None  # or set to a default Path if needed
            
            device = getattr(c, "device", None)

            from .modeling.marugoto.transformer.helpers import categorical_crossval_
            categorical_crossval_(clini_table=Path(c.clini_table), 
                                  slide_table=Path(c.slide_table),
                                  feature_dir=Path(c.feature_dir),
                                  output_path=Path(c.output_dir),
                                  target_label=c.target_label,
                                  cat_labels=c.cat_labels,
                                  cont_labels=c.cont_labels,
                                  categories=c.categories,
                                  loss_type=c.loss_type,
                                  n_splits=c.n_splits,
                                  model_type=c.model_type,
                                  use_coords=c.use_coords,
                                  bag_size=c.bag_size,
                                  checkpoint=checkpoint,
                                  device=device)
        case "deploy":
            require_configs(
                cfg,
                ["d_clini_table", "d_slide_table", "d_output_dir", "d_feature_dir", "target_label", "cat_labels", "cont_labels", "model_path"], # this one requires the model_path key!
                prefix="modeling",
                paths_to_check=["d_clini_table", "d_slide_table", "d_feature_dir"]
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import deploy_categorical_model_
            device = getattr(c, "device", None)
            deploy_categorical_model_(clini_table=Path(c.d_clini_table),
                                      slide_table=Path(c.d_slide_table),
                                      feature_dir=Path(c.d_feature_dir),
                                      output_path=Path(c.d_output_dir),
                                      target_label=c.target_label,
                                      cat_labels=c.cat_labels,
                                      cont_labels=c.cont_labels,
                                      model_path=Path(c.model_path),
                                      use_coords=c.use_coords,
                                      device=device)
            print("Successfully deployed models")

        case "statistics":
            require_configs(
                cfg,
                ["pred_csvs", "target_label", "output_dir"],
                prefix="modeling.statistics",
                paths_to_check=["pred_csvs"]
            )
            from .modeling.statistics import compute_stats
            c = cfg.modeling.statistics
            if isinstance(c.pred_csvs,str):
                c.pred_csvs = [c.pred_csvs]
            compute_stats(pred_csvs=[Path(x) for x in c.pred_csvs],
                          target_label=c.target_label,
                          true_class=c.true_class,
                          output_dir=Path(c.output_dir))
            print("Successfully calculated statistics")

        case "heatmaps":
            require_configs(
                cfg,
                ["feature_dir","wsi_dir","model_path","output_dir", "n_toptiles", "overview"], 
                prefix="heatmaps",
                paths_to_check=["feature_dir","wsi_dir","model_path"]
            )
            c = cfg.heatmaps
            from .heatmaps.__main__ import main
            main(slide_name=c.slide_name,
                 feature_dir=Path(c.feature_dir),
                 wsi_dir=Path(c.wsi_dir),
                 model_path=Path(c.model_path),
                 output_dir=Path(c.output_dir),
                 n_toptiles=int(c.n_toptiles),
                 overview=c.overview)
            print("Successfully produced heatmaps")

        
        case "preprocess_ddp":
            require_configs(
                cfg,
                ["output_dir", "wsi_dir", "cache_dir", "microns", "cores", "norm",
                 "del_slide", "only_feature_extraction", "device", "feat_extractor"],
                prefix="preprocessing",
                paths_to_check=["wsi_dir"],
            )
            c = cfg.preprocessing

            # Some checks
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            if c.norm and not Path(normalization_template_path).exists():
                raise ConfigurationError(f"Normalization template {normalization_template_path} does not exist, please run `stamp setup` to download it.")
            if c.feat_extractor == 'ctp':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/ctranspath.pth"
            elif c.feat_extractor == 'uni':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/uni/pytorch_model.bin"
            elif c.feat_extractor == 'uni2':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/uni2/pytorch_model.bin"
            elif c.feat_extractor == 'virchow2':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/virchow2/pytorch_model.bin"
            elif c.feat_extractor == 'h-optimus-0':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/h-optimus-0/pytorch_model.bin"
            elif c.feat_extractor == 'prov-gigapath':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/prov-gigapath/pytorch_model.bin"
            elif c.feat_extractor == 'ViT-B16':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/ViT-B16/pytorch_model.bin"
            else:
                raise ConfigurationError(f"Unknown feature extractor model: {c.feat_extractor}")
                
            if not Path(model_path).exists():
                raise ConfigurationError(f"Feature extractor model {model_path} does not exist, please run `stamp setup` to download it.")
            
            # handle single int or list[int] transparently
            from omegaconf.listconfig import ListConfig        # already imported above
            if isinstance(c.microns, (ListConfig, list, tuple)):
                # convert ListConfig → plain Python list so each element is an int
                scales = list(c.microns)
            else:
                scales = [c.microns]

            for m in scales:
                # make an isolated copy of the full config
                cfg_c = copy.deepcopy(cfg)
                cfg_c.preprocessing.microns = m
                # put every scale in its own sub-directory
                cfg_c.preprocessing.output_dir = (
                    Path(c.output_dir) / f"m{m}"
                )

                # re-use the same weights file that was resolved above
                from .preprocessing.wsi_norm_ddp import run_ddp_feature_extraction
                run_ddp_feature_extraction(cfg_c, model_path)

            print("Successfully preprocessed whole-slide images with ddp")


        case _:
            raise ConfigurationError(f"Unknown command {args.command}")
        


def main() -> None:
    parser = argparse.ArgumentParser(prog="stamp", description="STAMP: Solid Tumor Associative Modeling in Pathology")
    parser.add_argument("--config", "-c", type=Path, default=None, help=f"Path to config file (if unspecified, defaults to {DEFAULT_CONFIG_FILE.absolute()} or the default STAMP config file shipped with the package if {DEFAULT_CONFIG_FILE.absolute()} does not exist)")

    commands = parser.add_subparsers(dest="command")
    commands.add_parser("init", help="Create a new STAMP configuration file at the path specified by --config")
    commands.add_parser("setup", help="Download required resources")
    commands.add_parser("preprocess", help="Preprocess whole-slide images into feature vectors")
    commands.add_parser("train", help="Train a Vision Transformer model")
    commands.add_parser("crossval", help="Train a Vision Transformer model with cross validation for modeling.n_splits folds")
    commands.add_parser("deploy", help="Deploy a trained Vision Transformer model")
    commands.add_parser("statistics", help="Generate AUROCs and AUPRCs with 95%%CI for a trained Vision Transformer model")
    commands.add_parser("config", help="Print the loaded configuration")
    commands.add_parser("heatmaps", help="Generate heatmaps for a trained model")
    commands.add_parser("preprocess_ddp", help="Preprocess whole-slide images into feature vectors with DDP")
    args = parser.parse_args()

    # If no command is given, print help and exit
    if args.command is None:
        parser.print_help()
        exit(1)

    # Run the CLI
    try:
        run_cli(args)
    except ConfigurationError as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()
