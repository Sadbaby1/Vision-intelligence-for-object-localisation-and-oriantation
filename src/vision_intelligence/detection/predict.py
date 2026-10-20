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
    <out>/localisation.csv  per detection: depth method, X/Y/Z, size check, frame check, yaw (+ board frame)
    <out>/predictions.json  every detection with all localisation fields

3D localisation (geometry/localize.py) needs each image's depth and intrinsics; they are found
through the index.csv of the model's training dataset (data/datasets/<id>/index.csv).
"""

import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from vision_intelligence import paths
from vision_intelligence.detection import dataset
from vision_intelligence.detection.live import draw
from vision_intelligence.geometry.localize import read_intrinsics
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


def depth_lookup(weights):
    """
    {image stem: (depth path, intrinsics)} from the index.csv of the model's training dataset
    (data/datasets/<id>/index.csv), so test images get their aligned depth. {} if not available.
    """
    data = dataset.run_args(weights).get("data")
    index = Path(data).parent.parent / "index.csv" if data else None
    if index is None or not index.exists():
        return {}
    lookup, intrinsics = {}, {}
    with index.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (row.get("depth_path") and row.get("intrinsics_path")):
                continue
            ipath = row["intrinsics_path"]
            if ipath not in intrinsics:
                intrinsics[ipath] = read_intrinsics(paths.from_root(ipath))
            lookup[Path(row["name"]).stem] = (paths.from_root(row["depth_path"]), intrinsics[ipath])
    return lookup


def _output_dir(weights, source):
    """model/<run>/test_<name>; for a Roboflow-style <split>/images folder the split name is used."""
    name = source.stem if source.is_file() else (source.parent.name if source.name == "images" else source.name)
    return weights.parent.parent / f"test_{name}"


LOCALISATION_COLUMNS = ["image", "class", "confidence", "method", "valid_depth", "x_mm", "y_mm", "z_mm",
                        "length_mm", "width_mm", "size_ok", "wall_height_measured_mm", "frame_ok", "yaw_deg",
                        "board_x_mm", "board_y_mm", "height_mm", "yaw_board_deg"]


def _localisation_row(image_name, d):
    xyz = d.get("xyz_mm") or (None, None, None)
    bxyz = d.get("xyz_board_mm") or (None, None, None)
    size = d.get("size_mm") or (None, None)
    return {"image": image_name, "class": d["class"], "confidence": round(d["confidence"], 3), "method": d["method"],
            "valid_depth": d["valid_depth"], "x_mm": xyz[0], "y_mm": xyz[1], "z_mm": xyz[2],
            "length_mm": size[0], "width_mm": size[1], "size_ok": d.get("size_ok"),
            "wall_height_measured_mm": d.get("wall_height_measured_mm"), "frame_ok": d.get("frame_ok"),
            "yaw_deg": round(d["yaw_deg"], 1), "board_x_mm": bxyz[0], "board_y_mm": bxyz[1],
            "height_mm": d.get("height_mm"), "yaw_board_deg": d.get("yaw_board_deg")}


def _json_ready(d):
    out = {k: v for k, v in d.items() if k != "corners"}
    out["corners_px"] = np.asarray(d["corners"]).round(1).tolist()
    out["confidence"] = round(d["confidence"], 4)
    out["yaw_deg"] = round(d["yaw_deg"], 1)
    return out


def run(weights=None, source=None, conf=0.5, imgsz=None, out=None, iou_thresh=0.5, device=None, run_name=None):
    """
    Model: ``weights``, else ``run_name`` ('<dataset>/<run>'), else the latest trained run.
    ``source`` defaults to the test split of the model's training dataset and ``imgsz`` to the
    training image size, both recorded with the run.
    """
    from vision_intelligence.detection.live import PartDetector

    detector = PartDetector(weights, run_name, conf, imgsz, device)
    weights = detector.weights
    source = Path(source) if source else default_source(weights)
    images = [source] if source.is_file() else dataset.list_images(source)
    if not images:
        raise FileNotFoundError(f"no images in {source}")
    out = Path(out) if out else _output_dir(weights, source)
    out.mkdir(parents=True, exist_ok=True)

    names = {int(k): v for k, v in detector.model.names.items()}
    depths = depth_lookup(weights)
    print(f"{len(images)} images from {paths.to_root_relative(source)}; depth available for "
          f"{sum(p.stem in depths for p in images)} of them")
    rows, loc_rows, all_dets, tiles = [], [], {}, []
    for number, image_path in enumerate(images, 1):
        image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
        bgr = image[:, :, :3].copy()  # 4-channel RGB-D inputs are drawn on their colour part
        h, w = bgr.shape[:2]

        depth_path, intrinsics = depths.get(image_path.stem, (None, None))
        depth = cv.imread(str(depth_path), cv.IMREAD_UNCHANGED).astype(np.float32) if depth_path else None
        ready = image if image.ndim == 3 and image.shape[2] == detector.channels else None
        dets = detector(bgr, depth, intrinsics, image=ready)

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
        loc_rows += [_localisation_row(image_path.name, d) for d in dets]
        all_dets[image_path.name] = [_json_ready(d) for d in dets]
        print(summary.replace(f"[{number}]", f"[{number:2d}] {image_path.name}:"))

    cv.imwrite(str(out / "overview.png"), tile_overview(tiles))
    columns = (["tile", "image"] + [f"gt_{n}" for n in names.values()] + [f"pred_{n}" for n in names.values()]
               + ["tp", "fp", "fn", "status"])
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "localisation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOCALISATION_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(loc_rows)
    (out / "predictions.json").write_text(json.dumps(all_dets, indent=2), encoding="utf-8")

    labelled = [r for r in rows if "tp" in r]
    if labelled:
        tp, fp, fn = (sum(r[k] for r in labelled) for k in ("tp", "fp", "fn"))
        check = [r["tile"] for r in labelled if r["status"] == "CHECK"]
        print(f"\nTotal: {tp} correct, {fn} missed, {fp} false detections "
              f"(IoU >= {iou_thresh}, conf >= {conf}); images to check: {check or 'none'}")
    if loc_rows:
        located = [r for r in loc_rows if r["z_mm"] is not None]
        print(f"Localised {len(located)} of {len(loc_rows)} detections; size check ok for "
              f"{sum(bool(r['size_ok']) for r in located)}")
    print(f"Results in {out}")
    return rows
