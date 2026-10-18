"""
Visual test of a trained model on a folder of images (default: the test split of the
dataset the model was trained on, at its training image size). Class names come from the
model, so any class scheme works.

For every image the predictions are drawn together with the ground-truth labels (if a
label file exists) and matched to them, so you can see what the model found, missed
and added by mistake before trying it live on the camera.

    <out>/<image>.png       predictions (one colour per class) + ground truth (green) + missed (red)
    <out>/overview.png      all images as numbered tiles
    <out>/results.csv       per image: ground-truth and predicted counts, TP / FP / FN, status
    <out>/predictions.json  every detection: class, confidence, corners, approx. 3D centre, yaw

3D centres need the image's depth and intrinsics, found through ``data/datasets/index.csv``.
"""

import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from vision_intelligence import paths
from vision_intelligence.detection import dataset
from vision_intelligence.detection.live import detections, draw
from vision_intelligence.visualization import put_text_lines, tile_overview

GT_COLOR = (0, 200, 0)
MISSED_COLOR = (0, 0, 255)


def default_source(weights):
    """The test split of the dataset the model was trained on (from the run's args.yaml)."""
    data = dataset.run_args(weights).get("data")
    folder = dataset.split_dir(data, "test") if data and Path(data).exists() else None
    if folder is None or not folder.is_dir():
        raise FileNotFoundError(f"cannot find the test split of {weights}'s dataset ({data}); pass --source")
    return folder


def label_path_for(image_path):
    """
    YOLO convention: the last 'images' folder in the path becomes 'labels':
    .../images/<split>/x.png -> .../labels/<split>/x.txt   or   .../<split>/images/x.jpg -> .../<split>/labels/x.txt
    """
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_ground_truth(label_path, width, height, names):
    """
    List of (class name, 4x2 pixel corners) from a YOLO-OBB label file, or None if there is none.
    Class IDs are turned into names with ``names`` (the model's classes), so ground truth and
    predictions are compared by the same scheme.
    """
    if label_path is None or not label_path.exists():
        return None
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 9:
            if values:
                print(f"WARNING: {label_path}: skipping line with {len(values)} values (OBB needs 9)")
            continue
        cls_id = int(values[0])
        if cls_id not in names:
            print(f"WARNING: {label_path}: class {cls_id} is not one of the model's classes {names}")
        corners = np.array(values[1:], dtype=np.float32).reshape(4, 2) * [width, height]
        boxes.append((names.get(cls_id, str(cls_id)), corners.astype(np.float32)))
    return boxes


def polygon_iou(a, b):
    inter, _ = cv.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv.contourArea(a.astype(np.float32)) + cv.contourArea(b.astype(np.float32)) - inter
    return float(inter / union) if union > 0 else 0.0


def match(dets, gt, iou_thresh):
    """Greedy one-to-one matching by IoU within the same class. Returns (matched pred ids, missed gt boxes)."""
    pairs = sorted(
        ((polygon_iou(d["corners"], g[1]), di, gi)
         for di, d in enumerate(dets) for gi, g in enumerate(gt) if d["class"] == g[0]),
        reverse=True,
    )
    used_d, used_g = set(), set()
    for iou, di, gi in pairs:
        if iou < iou_thresh:
            break
        if di not in used_d and gi not in used_g:
            used_d.add(di)
            used_g.add(gi)
    return used_d, [g for gi, g in enumerate(gt) if gi not in used_g]


def _index_frames():
    index = paths.DATASETS / "index.csv"
    if not index.exists():
        return {}
    from vision_intelligence.camera.session import frames_from_index
    return frames_from_index(index)


def _output_dir(weights, source):
    """model/<run>/test_<name>; for a Roboflow-style <split>/images folder the split name is used."""
    name = source.stem if source.is_file() else (source.parent.name if source.name == "images" else source.name)
    return weights.parent.parent / f"test_{name}"


def run(weights=None, source=None, conf=0.5, imgsz=None, out=None, iou_thresh=0.5, device=None, run_name=None):
    """
    Model: ``weights``, else ``run_name`` ('<dataset>/<run>'), else the latest trained run.
    ``source`` defaults to the test split of the model's training dataset and ``imgsz`` to the
    training image size, both recorded with the run.
    """
    weights = dataset.resolve_weights(run_name, weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found; train a model first")
    source = Path(source) if source else default_source(weights)
    imgsz = imgsz or int(dataset.run_args(weights).get("imgsz", 640))
    images = [source] if source.is_file() else dataset.list_images(source)
    if not images:
        raise FileNotFoundError(f"no images in {source}")
    out = Path(out) if out else _output_dir(weights, source)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}
    print(f"Model {weights} (classes {names}), {len(images)} images from {source}, imgsz {imgsz}")
    frames = _index_frames()
    rows, all_dets, tiles = [], {}, []
    for number, image_path in enumerate(images, 1):
        image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
        bgr = image[:, :, :3].copy()  # 4-channel RGB-D inputs are drawn on their colour part
        h, w = bgr.shape[:2]

        frame = frames.get(f"{image_path.stem}.png")
        depth = cv.imread(str(frame.depth_path), cv.IMREAD_UNCHANGED).astype(np.float32) if frame else None
        pred = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        dets = detections(pred, depth, frame.intrinsics if frame else None)

        gt = load_ground_truth(label_path_for(image_path), w, h, names)
        annotated = bgr
        if gt is not None:
            for _, corners in gt:
                cv.polylines(annotated, [corners.astype(np.int32)], True, GT_COLOR, 2)
        annotated = draw(annotated, dets)

        row = {"tile": number, "image": image_path.name}
        row.update({f"pred_{n}": sum(d["class"] == n for d in dets) for n in names.values()})
        if gt is None:
            counts = "  ".join(f"{n} {row[f'pred_{n}']}" for n in names.values())
            summary = f"[{number}] no ground truth   {counts}"
            row.update(status="no labels")
        else:
            matched, missed = match(dets, gt, iou_thresh)
            for cls, corners in missed:
                cv.polylines(annotated, [corners.astype(np.int32)], True, MISSED_COLOR, 3)
                x, y = (int(v) for v in corners.min(axis=0))
                cv.putText(annotated, f"MISSED {cls}", (x, y + 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, MISSED_COLOR, 2)
            tp, fp, fn = len(matched), len(dets) - len(matched), len(missed)
            row.update({f"gt_{n}": sum(g[0] == n for g in gt) for n in names.values()})
            row.update(tp=tp, fp=fp, fn=fn, status="OK" if fp == 0 and fn == 0 else "CHECK")
            summary = f"[{number}] {row['status']}   correct {tp}   missed {fn}   false {fp}"
        put_text_lines(annotated, [summary, "green = ground truth, red = missed"])

        cv.imwrite(str(out / f"{image_path.stem}.png"), annotated)
        tiles.append(annotated)
        rows.append(row)
        all_dets[image_path.name] = [
            {"class": d["class"], "confidence": round(d["confidence"], 4),
             "corners_px": d["corners"].round(1).tolist(),
             "xyz_mm": [round(v, 1) for v in d["xyz_mm"]] if d["xyz_mm"] else None,
             "yaw_deg": round(d["yaw_deg"], 1)} for d in dets
        ]
        print(summary.replace(f"[{number}]", f"[{number:2d}] {image_path.name}:"))

    cv.imwrite(str(out / "overview.png"), tile_overview(tiles))
    columns = (["tile", "image"] + [f"gt_{n}" for n in names.values()] + [f"pred_{n}" for n in names.values()]
               + ["tp", "fp", "fn", "status"])
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    (out / "predictions.json").write_text(json.dumps(all_dets, indent=2), encoding="utf-8")

    labelled = [r for r in rows if "tp" in r]
    if labelled:
        tp, fp, fn = (sum(r[k] for r in labelled) for k in ("tp", "fp", "fn"))
        check = [r["tile"] for r in labelled if r["status"] == "CHECK"]
        print(f"\nTotal: {tp} correct, {fn} missed, {fp} false detections "
              f"(IoU >= {iou_thresh}, conf >= {conf}); images to check: {check or 'none'}")
    print(f"Results in {out}")
    return rows
