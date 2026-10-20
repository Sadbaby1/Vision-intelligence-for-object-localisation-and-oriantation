"""
``python -m vision_intelligence``               live camera view (same as ``vision-intelligence``)
``python -m vision_intelligence prepare-label`` select capture frames for labelling
``python -m vision_intelligence auto-label``    automatic YOLO-OBB labels for data/datasets/to_label
``python -m vision_intelligence split``         re-split a labelled dataset by time blocks per session
``python -m vision_intelligence train``         train YOLO-OBB (rgb or rgbd), weights saved under model/
``python -m vision_intelligence evaluate``      scores on test / test_white for all runs -> model/comparison.csv
``python -m vision_intelligence test``          predictions vs ground truth drawn on the test images
``python -m vision_intelligence detect``        live detection on the Orbbec camera with a trained model

Default paths are inside the project root (see ``paths.py``), so the commands work
from any folder. Paths given on the command line are relative to the current folder.
"""

import argparse
import dataclasses
import importlib.util
from pathlib import Path

from vision_intelligence import paths

_PREP = "vision_intelligence.detection."
_SESSION = "vision_intelligence.camera.session"
PREP_MODULES = {
    "prepare-label": [_PREP + "prepare_labeling", _SESSION],
    "auto-label": [_PREP + "auto_label", _SESSION],
    "pair-depth": [_PREP + "pair_depth", _SESSION],
    "split": [_PREP + "split_dataset"],
    "build-rgbd": [_PREP + "build_rgbd_dataset"],
}


def main():
    parser = argparse.ArgumentParser(prog="python -m vision_intelligence",
                                     epilog=f"project root: {paths.ROOT}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("view", help="live camera view (default)")

    p = sub.add_parser("prepare-label", help="copy distinct capture frames to data/datasets/to_label")
    p.add_argument("--captures", type=Path, default=paths.CAPTURES)
    p.add_argument("--out", type=Path, default=paths.DATASETS)
    p.add_argument("--source-folders", nargs="+", default=["filters", "frames"],
                   help="capture folders to read (not object classes)")
    p.add_argument("--min-gap", type=float, default=1.5,
                   help="frames saved less than this many seconds apart count as one shot (default 1.5)")
    p.add_argument("--every", type=int, default=1, help="keep every Nth shot (default 1 = all shots)")

    try:
        from vision_intelligence.detection.auto_label import Config
    except ImportError as e:  # e.g. module not in this checkout
        Config = None
        print(f"NOTE: auto-label unavailable ({e})")
    if Config is not None:
        p = sub.add_parser("auto-label", help="automatic YOLO-OBB labels from depth and colour")
        p.add_argument("--datasets", type=Path, default=paths.DATASETS)
        p.add_argument("--part-dimensions", type=Path, default=paths.PART_DIMENSIONS)
        for f in dataclasses.fields(Config):
            p.add_argument(f"--{f.name.replace('_', '-')}", type=f.type, default=f.default,
                           help=f"default {f.default}")

    p = sub.add_parser("pair-depth", help="pair a labelled Roboflow export with the original colour + depth captures "
                                          "(unchanged images only)")
    p.add_argument("--data", type=Path, required=True, help="Roboflow export data.yaml")
    p.add_argument("--out", type=Path, required=True, help="new folder, e.g. data/roboflow/rgbd_<export name>")
    p.add_argument("--captures", type=Path, default=paths.CAPTURES)
    p.add_argument("--max-diff", type=float, default=3.0,
                   help="max mean pixel difference to count an export image as unchanged (default 3, JPEG noise)")

    p = sub.add_parser("split", help="create a dataset version: re-split labelled data by time blocks per session")
    p.add_argument("--data", type=Path, required=True,
                   help="input dataset YAML, e.g. data/roboflow/<export>/data.yaml")
    p.add_argument("--dataset", required=True, help="new dataset ID, e.g. 2026-10-20_layers_v1 -> data/datasets/<id>/")
    p.add_argument("--shares", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--gap", type=int, default=3, help="frames dropped at each block boundary (default 3)")
    p.add_argument("--overwrite", action="store_true", help="replace the dataset if it already exists")

    sub.add_parser("board-print", help="write the printable ChArUco board (config/charuco_*_A3.pdf/.png)")

    p = sub.add_parser("calibrate-board", help="measure the camera -> board pose with the camera, save config/board_pose.json")
    p.add_argument("--frames", type=int, default=30, help="good board detections to average (default 30)")

    p = sub.add_parser("build-rgbd",help="build <id>/rgbd/: same images + local height from depth (4 channels)")
    p.add_argument("--dataset", required=True, help="dataset ID with rgb/ and depth linked in index.csv")
    p.add_argument("--kernel-px", type=int, default=151, help="local height kernel, wider than a part (default 151)")
    p.add_argument("--max-height-mm", type=float, default=40.0, help="height shown as full white (default 40)")
    p.add_argument("--blur-px", type=int, default=9, help="depth smoothing before local height, 0 = off (default 9)")
    p.add_argument("--overwrite", action="store_true", help="rebuild an existing rgbd/")

    p = sub.add_parser("train", help="train YOLO-OBB; weights saved in model/<dataset>/<run>/, never overwritten")
    p.add_argument("--dataset", help="dataset ID in data/datasets/ (uses <id>/<variant>/data.yaml)")
    p.add_argument("--data", type=Path, help="any dataset YAML instead of --dataset (e.g. a Roboflow data.yaml)")
    p.add_argument("--variant", choices=["rgb", "rgbd"], required=True,
                   help="rgb = colour images (<id>/rgb/), rgbd = colour + local height from depth (<id>/rgbd/)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="one run per seed, e.g. --seeds 0 1 2")
    p.add_argument("--name", help="run name (default: <variant>_<model>_s<seed>); only with a single seed")
    p.add_argument("--overwrite", action="store_true", help="replace an existing run of the same name")
    p.add_argument("--model-root", type=Path, default=paths.MODEL)
    p.add_argument("--device", default=None,
                   help="GPU index, 'mps' or 'cpu' (default: NVIDIA GPU, else Apple GPU, else CPU)")
    # Unset = use COMMON in detection/train.py; change only for quick tests, never between rgb and rgbd
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)

    sub.add_parser("runs", help="list trained runs (model/registry.csv)")

    p = sub.add_parser("evaluate", help="score trained models on test (and test_white) -> model/<dataset>/comparison.csv")
    p.add_argument("--dataset", nargs="+", help="only runs of these dataset IDs (default: all)")
    p.add_argument("--runs", nargs="+", help="only these runs, '<dataset>/<run>' or run name")
    p.add_argument("--data", type=Path, help="dataset YAML to test on (default: each run's training dataset)")
    p.add_argument("--model-root", type=Path, default=paths.MODEL)
    p.add_argument("--device", default=None, help="GPU index, 'mps' or 'cpu' (default: automatic)")

    p = sub.add_parser("test", aliases=["predict"],
                       help="draw predictions vs ground truth on test images, list found/missed/false per image")
    p.add_argument("--run", help="'<dataset>/<run>' (default: the most recently trained run)")
    p.add_argument("--weights", type=Path, help="weights file instead of --run")
    p.add_argument("--source", type=Path,
                   help="image file or folder (default: the test split of the model's training dataset)")
    p.add_argument("--out", type=Path, help="output folder (default: model/<dataset>/<run>/test_<source name>)")
    p.add_argument("--conf", type=float, default=0.5, help="minimum confidence (default 0.5)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a prediction as correct (default 0.5)")
    p.add_argument("--imgsz", type=int, help="inference size (default: the model's training size)")
    p.add_argument("--device", default=None)

    for name, text in (("detect", "live detection + 3D localisation on the Orbbec camera (RGB or RGB-D model)"),
                       ("measure", "repeatability test: localise a static scene N times, mean +- std per part")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--run", help="'<dataset>/<run>' (default: the most recently trained run)")
        p.add_argument("--weights", type=Path, help="weights file instead of --run")
        p.add_argument("--conf", type=float, default=0.5, help="minimum confidence to show a detection (default 0.5)")
        p.add_argument("--imgsz", type=int, help="inference size (default: the model's training size)")
        p.add_argument("--stretch", action="store_true",
                       help="squash each frame to imgsz x imgsz like Roboflow's 'Stretch' resize; only if training used it")
        p.add_argument("--device", default=None, help="GPU index, 'mps' or 'cpu' (default: automatic)")
    p.add_argument("--frames", type=int, default=10, help="number of frames to capture (default 10)")
    sub.choices["detect"].add_argument("--no-board", action="store_true",
                                       help="ignore config/board_pose.json (camera frame only)")

    args = parser.parse_args()

    # Dataset-preparation modules are not tracked in git (they need the raw captures anyway)
    missing = [m for m in PREP_MODULES.get(args.command, []) if importlib.util.find_spec(m) is None]
    if missing:
        parser.exit(1, f"'{args.command}' needs {', '.join(missing)}: a dataset-preparation module that is not "
                       "part of this checkout. Training, evaluate, test and detect work without it.\n")

    if args.command == "board-print":
        from vision_intelligence.geometry.board import render_print_files
        for f in render_print_files():
            print(f"Wrote {paths.to_root_relative(f)}")
    elif args.command == "calibrate-board":
        from vision_intelligence.geometry.board import calibrate_live
        calibrate_live(args.frames)
    elif args.command == "build-rgbd":
        from vision_intelligence.detection.build_rgbd_dataset import build
        build(args.dataset, args.kernel_px, args.max_height_mm, args.blur_px, args.overwrite)
    elif args.command == "pair-depth":
        from vision_intelligence.detection.pair_depth import pair
        pair(args.data, args.out, args.captures, args.max_diff)
    elif args.command == "split":
        from vision_intelligence.detection.split_dataset import split
        split(args.data, args.dataset, tuple(args.shares), args.gap, args.overwrite)
    elif args.command == "runs":
        registry = paths.MODEL / "registry.csv"
        print(registry.read_text(encoding="utf-8") if registry.exists() else "no runs registered yet")
    elif args.command == "evaluate":
        from vision_intelligence.detection.evaluate import evaluate
        evaluate(args.model_root, args.dataset, args.runs, args.device, args.data)
    elif args.command in ("test", "predict"):
        from vision_intelligence.detection.predict import run as predict
        predict(args.weights, args.source, args.conf, args.imgsz, args.out, args.iou, args.device, args.run)
    elif args.command == "detect":
        from vision_intelligence.detection.live import run
        run(args.weights, args.conf, args.imgsz, args.device, args.stretch, args.run, board=not args.no_board)
    elif args.command == "measure":
        from vision_intelligence.detection.measure import measure
        measure(args.weights, args.run, args.frames, args.conf, args.imgsz, args.device, args.stretch)
    elif args.command == "train":
        from vision_intelligence.detection.train import train
        if args.name and len(args.seeds) > 1:
            parser.error("--name works with a single seed only")
        overrides = {k: getattr(args, k) for k in ("epochs", "imgsz", "batch") if getattr(args, k) is not None}
        for seed in args.seeds:
            train(args.variant, seed, args.dataset, args.data, args.name, args.overwrite, args.model_root,
                  device=args.device, **overrides)
    elif args.command == "prepare-label":
        from vision_intelligence.detection.prepare_labeling import prepare
        prepare(args.captures, args.out, args.source_folders, args.min_gap, args.every)
    elif args.command == "auto-label":
        from vision_intelligence.detection.auto_label import auto_label
        cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
        auto_label(args.datasets, args.part_dimensions, cfg)
    else:
        from vision_intelligence.app import main as view
        view()


if __name__ == "__main__":
    main()
