"""
Train YOLO-OBB on the RGB (3-channel) or RGB-D (4-channel) dataset.

Both variants use exactly the same settings (``COMMON``), labels, splits, seeds and
pretrained start; that is what makes the RGB vs RGB-D comparison valid.

Dataset, either:
    --data <dataset yaml>   any Ultralytics dataset YAML, e.g. a Roboflow export's data.yaml
                            (<root>/{train,valid,test}/{images,labels}/)
    (default)               data/datasets/<variant>/{images,labels}/{train,val,test[,test_white]}/,
                            for which data/datasets/<variant>.yaml (+ _white.yaml) is written here
Output:
    model/<variant>_s<seed>/weights/{best,last}.pt   (+ args.yaml: dataset and imgsz used,
                                                      read back by evaluate / test / detect)
"""

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from vision_intelligence import paths
from vision_intelligence.detection import dataset

DATASETS_ROOT = paths.DATASETS
MODEL_ROOT = paths.MODEL
PRETRAINED = MODEL_ROOT / "pretrained" / "yolo11n-obb.pt"  # downloaded on first use
CLASS_NAMES = dataset.CLASS_NAMES  # Air_Filter is the union of filter and frame, used for detection but not scoring
VARIANT_CHANNELS = {"rgb": 3, "rgbd": 4}

# Identical for both models. HSV augmentation is off in both because it cannot apply to 4 channels.
# imgsz should equal the dataset's image size (1280 for the 1280x720 Roboflow export).
COMMON = dict(epochs=200, imgsz=1280, batch=8, patience=30, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, exist_ok=True)


def write_dataset_yaml(variant, datasets_root=DATASETS_ROOT):
    """
    Write ``<variant>.yaml`` (and ``<variant>_white.yaml`` if ``images/test_white`` exists)
    into ``datasets_root`` for this repo's layout. Returns the path of the main YAML.
    """
    datasets_root = Path(datasets_root)
    root = datasets_root / variant
    if not (root / "images" / "train").is_dir():
        raise FileNotFoundError(f"{root / 'images' / 'train'} not found; build the {variant} dataset first, "
                                "or pass --data <dataset yaml>")

    config = {
        "path": root.resolve().as_posix(),  # absolute, so training works from any directory
        "train": "images/train",
        "val": "images/val",
        "channels": VARIANT_CHANNELS[variant],
        # class names written by `split` into <root>/data.yaml win over the default scheme
        "names": dataset.class_names(root / "data.yaml") if (root / "data.yaml").exists() else CLASS_NAMES,
    }
    if (root / "images" / "test").is_dir():
        config["test"] = "images/test"

    yaml_path = datasets_root / f"{variant}.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    if (root / "images" / "test_white").is_dir():
        (datasets_root / f"{variant}_white.yaml").write_text(
            yaml.safe_dump({**config, "test": "images/test_white"}, sort_keys=False), encoding="utf-8"
        )
    else:
        print(f"NOTE: {root / 'images' / 'test_white'} does not exist; {variant}_white.yaml not written")
    return yaml_path


def default_device():
    """NVIDIA GPU if available, else Apple-silicon GPU, else CPU."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def input_channels(weights):
    """Number of input channels of the first conv layer of a trained model."""
    return YOLO(str(weights)).model.model[0].conv.weight.shape[1]


def check_dataset(data_yaml, imgsz):
    """Print a dataset summary; fail early on missing folders, warn if imgsz != image size."""
    print(f"Dataset {data_yaml}")
    print(f"  classes: {dataset.class_names(data_yaml)}")
    for split in ("train", "val", "test"):
        folder = dataset.split_dir(data_yaml, split)
        if folder is None:
            if split != "test":
                raise ValueError(f"{data_yaml} has no '{split}' entry")
            print("  test : (none)")
            continue
        if not folder.is_dir():
            raise FileNotFoundError(f"{split} folder {folder} from {data_yaml} does not exist")
        print(f"  {split:<5}: {len(dataset.list_images(folder))} images in {folder}")
    size = dataset.image_size(data_yaml)
    if size != imgsz:
        print(f"WARNING: training images are {size} px on their long side but imgsz={imgsz}; "
              f"images will be {'up' if imgsz > size else 'down'}scaled. Usually imgsz should be {size}.")


def train(variant="rgb", seed=0, datasets_root=DATASETS_ROOT, model_root=MODEL_ROOT,
          pretrained=PRETRAINED, device=None, data=None, **overrides):
    """Train one model and return the path of its best weights."""
    data_yaml = Path(data).resolve() if data else write_dataset_yaml(variant, datasets_root)
    settings = {**COMMON, **overrides}
    check_dataset(data_yaml, settings["imgsz"])
    if device is None:
        device = default_device()
    if device == "cpu":
        print("WARNING: no GPU found, training on CPU will be slow")

    Path(pretrained).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pretrained))
    print(f"Training {variant} (seed {seed}) with {settings}")
    model.train(
        data=str(data_yaml),
        project=str(Path(model_root).resolve()),
        name=f"{variant}_s{seed}",
        seed=seed,
        deterministic=True,
        device=device,
        **settings,
    )

    best = Path(model.trainer.best)
    got, expected = input_channels(best), dataset.channels(data_yaml)
    print(f"First conv layer input channels: {got} (expected {expected})")
    if got != expected:
        raise RuntimeError(f"{best}: model takes {got} input channels, expected {expected}; "
                           "the extra channel was dropped, stop and investigate")
    print(f"Best weights: {best}")
    print("Next: python -m vision_intelligence evaluate   (scores)   |   "
          f"python -m vision_intelligence test --weights {paths.to_root_relative(best)}   (visual check)")
    return best
