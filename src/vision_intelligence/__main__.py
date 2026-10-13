"""
``python -m vision_intelligence``               live camera view (same as ``vision-intelligence``)
``python -m vision_intelligence prepare-label`` select capture frames for labelling
``python -m vision_intelligence auto-label``    automatic YOLO-OBB labels for data/datasets/to_label
``python -m vision_intelligence train``         train YOLO-OBB (rgb or rgbd), weights saved under model/
``python -m vision_intelligence detect``        live detection on the Orbbec camera with a trained model

Default paths are inside the project root (see ``paths.py``), so the commands work
from any folder. Paths given on the command line are relative to the current folder.
"""

import argparse
import dataclasses
from pathlib import Path

from vision_intelligence import paths


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

    p = sub.add_parser("train", help="train YOLO-OBB on the rgb or rgbd dataset (weights saved under model/)")
    p.add_argument("--variant", choices=["rgb", "rgbd"], default="rgb")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="one run per seed, e.g. --seeds 0 1 2")
    p.add_argument("--datasets", type=Path, default=paths.DATASETS)
    p.add_argument("--model-root", type=Path, default=paths.MODEL)
    p.add_argument("--device", default=None,
                   help="GPU index, 'mps' or 'cpu' (default: NVIDIA GPU, else Apple GPU, else CPU)")
    # Unset = use COMMON in detection/train.py; change only for quick tests, never between rgb and rgbd
    p.add_argument("--epochs", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)

    p = sub.add_parser("detect", help="live detection on the Orbbec camera with a trained model")
    p.add_argument("--weights", type=Path, default=paths.MODEL / "rgb_s0" / "weights" / "best.pt")
    p.add_argument("--conf", type=float, default=0.5, help="minimum confidence to show a detection (default 0.5)")
    p.add_argument("--imgsz", type=int, default=1280, help="inference size; keep equal to the training size")
    p.add_argument("--device", default=None, help="GPU index, 'mps' or 'cpu' (default: automatic)")

    args = parser.parse_args()

    if args.command == "detect":
        from vision_intelligence.detection.live import run
        run(args.weights, args.conf, args.imgsz, args.device)
    elif args.command == "train":
        from vision_intelligence.detection.train import train
        overrides = {k: getattr(args, k) for k in ("epochs", "imgsz", "batch") if getattr(args, k) is not None}
        for seed in args.seeds:
            train(args.variant, seed, args.datasets, args.model_root, device=args.device, **overrides)
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
