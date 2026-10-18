"""
Re-split a labelled YOLO dataset into train / val / test by contiguous time blocks per
capture session, so near-identical neighbouring frames never end up in both train and test
(METHODOLOGY.md, "Splits by session where possible").

Input: any Ultralytics dataset YAML (e.g. a Roboflow export's data.yaml). All images of
all its splits are pooled. Each image name must start with ``<session>_<frame number>``
(e.g. ``20261016_111506_0042``); Roboflow suffixes such as ``_png.rf.<hash>`` are ignored.

Per session, frames are sorted and split into blocks: first ``train`` share -> train,
next ``val`` share -> val, rest -> test. ``gap`` frames are dropped at each block boundary.

Augmented copies (several files of the same source frame, as Roboflow creates for train):
all copies are kept in train; val/test keep one copy, preferring one that came from the
export's own val/test split (those are not augmented).

Output (input files are only copied, never changed):
    <out>/images/{train,val,test}/, <out>/labels/{train,val,test}/
    <out>/data.yaml          class names from the input, absolute path
    <out>/split_report.csv   one row per source frame: session, frame, split, copies, origin
"""

import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from vision_intelligence import paths
from vision_intelligence.detection import dataset

NAME_PATTERN = re.compile(r"^(?P<session>.+)_(?P<frame>\d+)$")
SPLITS = ("train", "val", "test")


def source_key(image_path):
    """(session, frame) from a file name like 20261016_111506_0042_png.rf.abc123.jpg, or None."""
    base = image_path.stem.split(".rf.")[0]
    base = re.sub(r"_(png|jpe?g|bmp|tiff?)$", "", base, flags=re.IGNORECASE)
    m = NAME_PATTERN.match(base)
    return (m["session"], int(m["frame"])) if m else None


def collect(data_yaml):
    """{(session, frame): [(image, label, origin split), ...]} and a list of unparseable images."""
    from vision_intelligence.detection.predict import label_path_for

    sources, unparsed = defaultdict(list), []
    for origin in ("train", "val", "test"):
        folder = dataset.split_dir(data_yaml, origin)
        if folder is None:
            continue
        for image in dataset.list_images(folder):
            label = label_path_for(image)
            if label is None or not label.exists():
                print(f"WARNING: no label for {image}, skipping")
                continue
            key = source_key(image)
            if key is None:
                unparsed.append((image, label, origin))
            else:
                sources[key].append((image, label, origin))
    return sources, unparsed


def assign_blocks(frames, shares, gap):
    """Split a sorted list of frame numbers into contiguous blocks; returns {frame: split or None}."""
    n = len(frames)
    n_train = round(n * shares[0])
    n_val = round(n * shares[1])
    assignment = {}
    for i, frame in enumerate(frames):
        if i < n_train:
            split = "train"
        elif i < n_train + gap:
            split = None
        elif i < n_train + gap + n_val:
            split = "val"
        elif i < n_train + 2 * gap + n_val:
            split = None
        else:
            split = "test"
        assignment[frame] = split
    return assignment


def split(data, out=None, shares=(0.70, 0.15, 0.15), gap=3, overwrite=False):
    data = Path(data).resolve()
    out = Path(out) if out else paths.DATASETS / "rgb"
    if any((out / d).exists() and any((out / d).iterdir()) for d in ("images", "labels")):
        if not overwrite:
            raise FileExistsError(f"{out} already has images/labels; pass --overwrite to replace them")
        for d in ("images", "labels"):
            shutil.rmtree(out / d, ignore_errors=True)

    names = dataset.class_names(data)
    sources, unparsed = collect(data)
    if unparsed:
        print(f"WARNING: {len(unparsed)} image names do not start with <session>_<frame>; they go to train, "
              f"e.g. {unparsed[0][0].name}")

    by_session = defaultdict(list)
    for session, frame in sources:
        by_session[session].append(frame)

    report, placed, eval_augmented, dropped = [], Counter(), 0, 0
    instances = {s: Counter() for s in SPLITS}
    for split_name in SPLITS:
        (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    def place(files, split_name):
        for image, label, _ in files:
            shutil.copy2(image, out / "images" / split_name / image.name)
            shutil.copy2(label, out / "labels" / split_name / f"{image.stem}.txt")
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    instances[split_name][names.get(int(line.split()[0]), line.split()[0])] += 1
            placed[split_name] += 1

    print(f"{'session':<20} {'frames':>6}  train         val           test")
    for session in sorted(by_session):
        frames = sorted(by_session[session])
        assignment = assign_blocks(frames, shares, gap)
        ranges = {s: [f for f in frames if assignment[f] == s] for s in SPLITS}
        text = "  ".join(f"{r[0]:>4}-{r[-1]:<4} ({len(r):>3})" if r else f"{'-':>13}"
                         for r in (ranges[s] for s in SPLITS))
        print(f"{session:<20} {len(frames):>6}  {text}")
        for frame in frames:
            split_name = assignment[frame]
            copies = sources[(session, frame)]
            if split_name is None:
                dropped += 1
                report.append({"session": session, "frame": frame, "split": "dropped (gap)",
                               "copies": len(copies), "origin": "+".join(sorted({c[2] for c in copies}))})
                continue
            if split_name != "train":
                clean = [c for c in copies if c[2] != "train"]
                if not clean:
                    eval_augmented += 1
                copies = (clean or copies)[:1]
            place(copies, split_name)
            report.append({"session": session, "frame": frame, "split": split_name,
                           "copies": len(copies), "origin": "+".join(sorted({c[2] for c in copies}))})
    place(unparsed, "train")

    config = {"path": out.resolve().as_posix(), "train": "images/train", "val": "images/val",
              "test": "images/test", "channels": dataset.channels(data), "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (out / "split_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session", "frame", "split", "copies", "origin"])
        writer.writeheader()
        writer.writerows(report)

    print(f"\nImages: " + ", ".join(f"{s} {placed[s]}" for s in SPLITS) + f"; {dropped} source frames dropped as gaps")
    for s in SPLITS:
        print(f"  {s:<5} instances: {dict(instances[s])}")
    missing = [(s, n) for s in ("val", "test") for n in names.values() if instances[s][n] == 0]
    if missing:
        print(f"WARNING: classes with no instances: {missing}")
    if eval_augmented:
        print(f"WARNING: {eval_augmented} val/test frames exist only as augmented copies (from the export's train "
              "split); their boxes may be loose. Re-export without augmentation for a clean test set.")
    print(f"Wrote {paths.to_root_relative(out)} (data.yaml, split_report.csv). "
          f"Train with: python -m vision_intelligence train --data {paths.to_root_relative(out / 'data.yaml')}")
    return out / "data.yaml"
