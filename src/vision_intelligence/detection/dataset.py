"""
Helpers for datasets and training runs, shared by split / train / evaluate / test / detect,
so they all agree on folders, classes and image size.

Project layout (one folder per dataset version, models grouped by the dataset they were trained on):

    data/captures/<folder>/<session>/          raw camera data: color/, depth/, intrinsics.json
    data/roboflow/<export>/                    raw Roboflow exports, as downloaded
    data/datasets/<dataset_id>/
        manifest.json                          where the data came from, split settings, classes
        index.csv                              every image -> session, frame, split, source files
        rgb/  {images,labels}/{train,val,test[,test_white]}/ + data.yaml
        rgbd/ ... (4-channel images)                         + data.yaml
    model/<dataset_id>/<variant>_<model>_s<seed>/
        weights/best.pt, args.yaml (Ultralytics), run_info.json (ours)
    model/<dataset_id>/comparison.csv          evaluate results for the models of that dataset
    model/registry.csv                         one row per training run

Dataset YAMLs may use this repo's layout (images/<split>, labels/<split>) or a Roboflow
export's (<split>/images, <split>/labels). YAMLs written here have no absolute ``path``,
so folders are relative to the YAML and keep working after the repo is moved.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2 as cv
import yaml

from vision_intelligence import paths

# Class scheme of the current dataset (0 is the assembled filter-in-frame product)
CLASS_NAMES = {0: "Air_Filter", 1: "filter", 2: "frame"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VARIANTS = ("rgb", "rgbd")
REGISTRY_COLUMNS = ["run", "dataset", "variant", "model", "seed", "channels", "imgsz", "epochs_run",
                    "best_epoch", "val_mAP50", "val_mAP50-95", "created"]


# ---------------------------------------------------------------- dataset YAMLs

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def write_data_yaml(path, names, channels=3, splits=("train", "val", "test")):
    """Dataset YAML for this repo's layout, relative to the YAML's own folder (no absolute path)."""
    config = {s: f"images/{s}" for s in splits}
    config.update(channels=int(channels), names={int(k): v for k, v in names.items()})
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return Path(path)


def class_names(data_yaml):
    """{id: name} from a dataset YAML (``names`` may be a list or a dict)."""
    names = load_yaml(data_yaml).get("names", CLASS_NAMES)
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def split_dir(data_yaml, split):
    """
    Image folder of ``split`` ('train', 'val', 'test', ...) resolved like Ultralytics does:
    relative to ``path`` if given, else to the YAML's folder; a leading '../' is also tried without it.
    Returns None if the YAML has no such split.
    """
    data_yaml = Path(data_yaml).resolve()
    cfg = load_yaml(data_yaml)
    entry = cfg.get(split)
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    if not entry:
        return None
    base = Path(cfg["path"]) if cfg.get("path") else data_yaml.parent
    if not base.is_absolute():
        base = data_yaml.parent / base
    folder = (base / entry).resolve()
    if not folder.exists() and str(entry).startswith("../"):
        folder = (base / str(entry)[3:]).resolve()
    return folder


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if folder.is_dir() else []


def channels(data_yaml):
    return int(load_yaml(data_yaml).get("channels", 3))


def image_size(data_yaml):
    """Longest side of the first training image: the natural ``imgsz`` for this dataset."""
    images = list_images(split_dir(data_yaml, "train"))
    if not images:
        raise FileNotFoundError(f"no training images found for {data_yaml}")
    img = cv.imread(str(images[0]), cv.IMREAD_UNCHANGED)
    return int(max(img.shape[:2]))


# ---------------------------------------------------------------- dataset folders

def dataset_dir(dataset_id):
    return paths.DATASETS / dataset_id


def variant_yaml(dataset_id, variant):
    """data/datasets/<id>/<variant>/data.yaml (raises with a hint if it does not exist)."""
    path = dataset_dir(dataset_id) / variant / "data.yaml"
    if not path.exists():
        hint = "run `split`" if variant == "rgb" else "build the RGB-D dataset first"
        raise FileNotFoundError(f"{path} not found; {hint}")
    return path


def dataset_id_for(data_yaml):
    """Dataset ID of a YAML inside data/datasets/<id>/..., else the name of the YAML's folder."""
    data_yaml = Path(data_yaml).resolve()
    try:
        return data_yaml.relative_to(paths.DATASETS.resolve()).parts[0]
    except ValueError:
        return data_yaml.parent.name


def write_manifest(dataset_id, **info):
    path = dataset_dir(dataset_id) / "manifest.json"
    data = {"dataset_id": dataset_id, "created": datetime.now().isoformat(timespec="seconds"), **info}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- training runs

def run_name(variant, model_name, seed):
    return f"{variant}_{model_name}_s{seed}"


def parse_run_name(name):
    """('rgb', 'yolo11n-obb', 0) from 'rgb_yolo11n-obb_s0'; unknown parts become None."""
    head, _, seed = name.rpartition("_s")
    variant, _, model_name = head.partition("_")
    return (variant or None, model_name or None, int(seed) if seed.isdigit() else None)


def find_runs(model_root=paths.MODEL):
    """All trained runs model/<dataset>/<run>/weights/best.pt as (run_dir, dataset_id), sorted."""
    root = Path(model_root)
    runs = [(w.parent.parent, w.parent.parent.parent.name) for w in root.glob("*/*/weights/best.pt")]
    return sorted(runs, key=lambda r: (r[1], r[0].name))


def latest_weights(model_root=paths.MODEL):
    """best.pt of the most recently trained run, or None."""
    weights = list(Path(model_root).glob("*/*/weights/best.pt"))
    return max(weights, key=lambda p: p.stat().st_mtime) if weights else None


def resolve_weights(run=None, weights=None, model_root=paths.MODEL):
    """
    Weights to use: ``weights`` if given; else ``run`` ('<dataset>/<run>', or a run name if it
    is unique); else the most recently trained run. Prints which one is used.
    """
    if weights:
        return Path(weights)
    if run:
        candidates = [Path(model_root) / run / "weights" / "best.pt"]
        candidates += [r / "weights" / "best.pt" for r, _ in find_runs(model_root) if r.name == run]
        found = [c for c in dict.fromkeys(candidates) if c.exists()]
        if len(found) != 1:
            raise FileNotFoundError(f"run '{run}': {'not found' if not found else 'ambiguous'}; "
                                    f"use <dataset>/<run>, available: {[f'{d}/{r.name}' for r, d in find_runs(model_root)]}")
        chosen = found[0]
    else:
        chosen = latest_weights(model_root)
        if chosen is None:
            raise FileNotFoundError(f"no trained runs in {model_root}; train a model first")
    print(f"Using model {paths.to_root_relative(chosen)}")
    return chosen


def write_run_info(run_dir, **info):
    path = Path(run_dir) / "run_info.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


def run_args(weights):
    """
    Training settings of the run that produced ``weights``: Ultralytics' ``args.yaml``,
    with the dataset path taken from our ``run_info.json`` (project-relative) when present.
    """
    run_dir = Path(weights).resolve().parent.parent
    args = load_yaml(run_dir / "args.yaml") if (run_dir / "args.yaml").exists() else {}
    info_file = run_dir / "run_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if info.get("data"):
            args["data"] = str(paths.from_root(info["data"]))
        if info.get("imgsz"):
            args["imgsz"] = info["imgsz"]
    return args


def append_registry(row, registry=None):
    """Add or replace one run's row in model/registry.csv."""
    registry = Path(registry or paths.MODEL / "registry.csv")
    rows = []
    if registry.exists():
        with registry.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run") != row["run"]]
    rows.append(row)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["run"]))
    return registry


def best_val_metrics(run_dir):
    """(epochs run, best epoch, mAP50, mAP50-95) of the best validation epoch in results.csv."""
    results = Path(run_dir) / "results.csv"
    if not results.exists():
        return None, None, None, None
    with results.open(newline="", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return 0, None, None, None
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return len(rows), int(float(best["epoch"])), round(float(best["metrics/mAP50(B)"]), 4), \
        round(float(best["metrics/mAP50-95(B)"]), 4)
