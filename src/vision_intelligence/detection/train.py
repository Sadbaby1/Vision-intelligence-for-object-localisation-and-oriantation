"""
Train YOLO-OBB on the RGB (3-channel) or RGB-D (4-channel) version of a dataset.

Both variants use exactly the same settings (``COMMON``), labels, splits, seeds and
pretrained start; that is what makes the RGB vs RGB-D comparison valid.

Dataset, either:
    --dataset <id> --variant rgb|rgbd   data/datasets/<id>/<variant>/data.yaml (written by `split` / RGB-D build)
    --data <dataset yaml>               any Ultralytics dataset YAML (e.g. a Roboflow export's data.yaml)
Output (an existing run is never overwritten unless --overwrite):
    model/<dataset_id>/<variant>_<model>_s<seed>/weights/{best,last}.pt
    model/<dataset_id>/<run>/run_info.json   dataset, classes, channels, imgsz, pretrained, date
    model/registry.csv                       one row per run
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from vision_intelligence import paths
from vision_intelligence.detection import dataset

MODEL_ROOT = paths.MODEL
PRETRAINED = MODEL_ROOT / "pretrained" / "yolo11n-obb.pt"  # downloaded on first use
CLASS_NAMES = dataset.CLASS_NAMES  # Air_Filter is the union of filter and frame, used for detection but not scoring

# Identical for both models. HSV augmentation is off in both because it cannot apply to 4 channels.
# imgsz should equal the dataset's image size (1280 for the 1280x720 Roboflow export).
COMMON = dict(epochs=200, imgsz=1280, batch=8, patience=30, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)


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
    print(f"Dataset {paths.to_root_relative(data_yaml)}")
    print(f"  classes: {dataset.class_names(data_yaml)}, channels: {dataset.channels(data_yaml)}")
    for split in ("train", "val", "test"):
        folder = dataset.split_dir(data_yaml, split)
        if folder is None:
            if split != "test":
                raise ValueError(f"{data_yaml} has no '{split}' entry")
            print("  test : (none)")
            continue
        if not folder.is_dir():
            raise FileNotFoundError(f"{split} folder {folder} from {data_yaml} does not exist")
        print(f"  {split:<5}: {len(dataset.list_images(folder))} images")
    size = dataset.image_size(data_yaml)
    if size != imgsz:
        print(f"WARNING: training images are {size} px on their long side but imgsz={imgsz}; "
              f"images will be {'up' if imgsz > size else 'down'}scaled. Usually imgsz should be {size}.")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def train(variant="rgb", seed=0, dataset_id=None, data=None, name=None, overwrite=False,
          model_root=MODEL_ROOT, pretrained=PRETRAINED, device=None, **overrides):
    """Train one model and return the path of its best weights."""
    if data:
        data_yaml = Path(data).resolve()
        dataset_id = dataset_id or dataset.dataset_id_for(data_yaml)
    elif dataset_id:
        data_yaml = dataset.variant_yaml(dataset_id, variant).resolve()
    else:
        raise ValueError("pass --dataset <id> (data/datasets/<id>/<variant>/) or --data <dataset yaml>")

    model_name = Path(pretrained).stem
    name = name or dataset.run_name(variant, model_name, seed)
    run_dir = Path(model_root) / dataset_id / name
    if (run_dir / "weights").exists() and not overwrite:
        raise FileExistsError(f"{paths.to_root_relative(run_dir)} already exists; choose another --name, "
                              "or pass --overwrite to replace it")

    settings = {**COMMON, **overrides}
    check_dataset(data_yaml, settings["imgsz"])
    expected = dataset.channels(data_yaml)
    if variant == "rgbd" and expected != 4:
        print(f"WARNING: variant rgbd but the dataset has channels: {expected}")
    if device is None:
        device = default_device()
    if device == "cpu":
        print("WARNING: no GPU found, training on CPU will be slow")

    Path(pretrained).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pretrained))
    print(f"Training {paths.to_root_relative(run_dir)} with {settings}")
    model.train(
        data=str(data_yaml),
        project=str((Path(model_root) / dataset_id).resolve()),
        name=name,
        exist_ok=True,  # the folder check above already protects existing runs
        seed=seed,
        deterministic=True,
        device=device,
        **settings,
    )

    best = Path(model.trainer.best)
    got = input_channels(best)
    print(f"First conv layer input channels: {got} (expected {expected})")
    if got != expected:
        raise RuntimeError(f"{best}: model takes {got} input channels, expected {expected}; "
                           "the extra channel was dropped, stop and investigate")

    epochs_run, best_epoch, map50, map50_95 = dataset.best_val_metrics(run_dir)
    created = datetime.now().isoformat(timespec="seconds")
    # RGB-D: keep the local-height settings with the model, so prediction can rebuild the same input
    rgbd_config_file = data_yaml.parent / "rgbd_config.json"
    rgbd_config = json.loads(rgbd_config_file.read_text(encoding="utf-8")) if rgbd_config_file.exists() else None
    dataset.write_run_info(
        run_dir, run=f"{dataset_id}/{name}", dataset=dataset_id, data=paths.to_root_relative(data_yaml),
        variant=variant, model=model_name, pretrained=paths.to_root_relative(pretrained), seed=seed,
        classes=dataset.class_names(data_yaml), channels=expected, imgsz=settings["imgsz"],
        settings=settings, epochs_run=epochs_run, best_epoch=best_epoch,
        val_mAP50=map50, val_mAP50_95=map50_95, git_commit=_git_commit(), created=created,
        rgbd_config=rgbd_config,
    )
    dataset.append_registry({
        "run": f"{dataset_id}/{name}", "dataset": dataset_id, "variant": variant, "model": model_name,
        "seed": seed, "channels": expected, "imgsz": settings["imgsz"], "epochs_run": epochs_run,
        "best_epoch": best_epoch, "val_mAP50": map50, "val_mAP50-95": map50_95, "created": created,
    })
    print(f"Best weights: {paths.to_root_relative(best)}  (registered in model/registry.csv)")
    print(f"Next: python -m vision_intelligence evaluate --dataset {dataset_id}   (scores)   |   "
          f"python -m vision_intelligence test --run {dataset_id}/{name}   (visual check)")
    return best
