"""
Train YOLO-OBB on the RGB (3-channel) or RGB-D (4-channel) dataset.

Both variants use exactly the same settings (``COMMON``), labels, splits, seeds and
pretrained start; that is what makes the RGB vs RGB-D comparison valid.

    data/datasets/rgb/{images,labels}/{train,val,test[,test_white]}/   (input)
    data/datasets/rgbd/...                                              (input, Phase 2)
    data/datasets/{rgb,rgbd}.yaml, {rgb,rgbd}_white.yaml               (written here)
    model/<variant>_s<seed>/weights/{best,last}.pt                      (output)
"""

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from vision_intelligence import paths

DATASETS_ROOT = paths.DATASETS
MODEL_ROOT = paths.MODEL
PRETRAINED = MODEL_ROOT / "pretrained" / "yolo11n-obb.pt"  # downloaded on first use
CLASS_NAMES = {0: "filter", 1: "frame"}
VARIANT_CHANNELS = {"rgb": 3, "rgbd": 4}

# Identical for both models. HSV augmentation is off in both because it cannot apply to 4 channels.
COMMON = dict(epochs=100, imgsz=1280, batch=8, patience=30, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, exist_ok=True)


def write_dataset_yaml(variant, datasets_root=DATASETS_ROOT):
    """
    Write ``<variant>.yaml`` (and ``<variant>_white.yaml`` if ``images/test_white`` exists)
    into ``datasets_root``. Returns the path of the main YAML.
    """
    datasets_root = Path(datasets_root)
    root = datasets_root / variant
    if not (root / "images" / "train").is_dir():
        raise FileNotFoundError(f"{root / 'images' / 'train'} not found; build the {variant} dataset first")

    config = {
        "path": root.resolve().as_posix(),  # absolute, so training works from any directory
        "train": "images/train",
        "val": "images/val",
        "channels": VARIANT_CHANNELS[variant],
        "names": CLASS_NAMES,
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


def train(variant="rgb", seed=0, datasets_root=DATASETS_ROOT, model_root=MODEL_ROOT,
          pretrained=PRETRAINED, device=None, **overrides):
    """Train one model and return the path of its best weights."""
    data_yaml = write_dataset_yaml(variant, datasets_root)
    if device is None:
        device = default_device()
    if device == "cpu":
        print("WARNING: no GPU found, training on CPU will be slow")

    Path(pretrained).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pretrained))
    settings = {**COMMON, **overrides}
    print(f"Training {variant} (seed {seed}) on {data_yaml} with {settings}")
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
    channels = input_channels(best)
    expected = VARIANT_CHANNELS[variant]
    print(f"First conv layer input channels: {channels} (expected {expected})")
    if channels != expected:
        raise RuntimeError(f"{best}: model takes {channels} input channels, expected {expected}; "
                           "the extra channel was dropped, stop and investigate")
    print(f"Best weights: {best}")
    return best
