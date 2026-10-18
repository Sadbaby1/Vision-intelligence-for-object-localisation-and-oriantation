"""
Helpers for Ultralytics dataset YAMLs and training runs, shared by train / evaluate /
test / live, so they all agree on classes, image folders and image size.

Supports both layouts:
    <root>/images/{train,val,test}/ + <root>/labels/...     (this repo's layout)
    <root>/{train,valid,test}/{images,labels}/ + data.yaml   (Roboflow export)
"""

from pathlib import Path

import cv2 as cv
import yaml

# Class scheme of the current dataset (0 is the assembled filter-in-frame product)
CLASS_NAMES = {0: "Air_Filter", 1: "filter", 2: "frame"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


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


def run_args(weights):
    """Training arguments Ultralytics saved next to the weights (``<run>/args.yaml``), or {}."""
    args_file = Path(weights).resolve().parent.parent / "args.yaml"
    return load_yaml(args_file) if args_file.exists() else {}
