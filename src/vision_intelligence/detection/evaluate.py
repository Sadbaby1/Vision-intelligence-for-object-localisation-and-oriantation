"""
Evaluate trained models on the held-out splits (METHODOLOGY.md Phase 4).

Every run ``model/<dataset_id>/<run>/`` is scored on the ``test`` split of the dataset it
was trained on (from its ``run_info.json`` / ``args.yaml``, or ``--data``), and on the
sibling ``<yaml stem>_white.yaml`` if it exists (filters on white backgrounds). Image size
is the one used for training. Results go to:

    model/<dataset_id>/<run>/eval_<subset>/   Ultralytics plots (confusion matrix, PR curves, predictions)
    model/<dataset_id>/comparison.csv         one row per run and subset, metrics for every class

Filter recall on ``test_white`` is the primary metric: a missed filter lowers recall.
"""

import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from vision_intelligence import paths
from vision_intelligence.detection import dataset
from vision_intelligence.detection.train import COMMON, MODEL_ROOT


def class_metrics(box, cls_id):
    """(precision, recall, AP50, AP50-95) of one class, or NaNs if it has no instances in the subset."""
    present = [int(c) for c in box.ap_class_index]
    if cls_id not in present:
        return (float("nan"),) * 4
    return tuple(float(v) for v in box.class_result(present.index(cls_id)))


def run_dataset(run_dir, data=None):
    """Dataset YAML for a run: --data, else the one recorded with the run."""
    if data:
        return Path(data).resolve()
    recorded = dataset.run_args(run_dir / "weights" / "best.pt").get("data")
    if recorded and Path(recorded).exists():
        return Path(recorded).resolve()
    raise FileNotFoundError(f"{run_dir}: training dataset {recorded} not found; pass --data")


def evaluate_run(run_dir, dataset_id, data=None, device=None):
    data_yaml = run_dataset(run_dir, data)
    subsets = {"test": data_yaml}
    white_yaml = data_yaml.with_name(f"{data_yaml.stem}_white.yaml")
    if white_yaml.exists():
        subsets["test_white"] = white_yaml

    weights = run_dir / "weights" / "best.pt"
    imgsz = int(dataset.run_args(weights).get("imgsz", COMMON["imgsz"]))
    variant, model_name, seed = dataset.parse_run_name(run_dir.name)
    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}

    rows = []
    for subset, yaml_path in subsets.items():
        folder = dataset.split_dir(yaml_path, "test")
        n_images = len(dataset.list_images(folder)) if folder else 0
        if n_images == 0:
            print(f"WARNING: no test images for {run_dir.name}/{subset} ({folder}), skipping")
            continue
        print(f"  {subset}: {n_images} images from {paths.to_root_relative(folder)}, imgsz {imgsz}")
        m = model.val(data=str(yaml_path), split="test", imgsz=imgsz, batch=COMMON["batch"],
                      project=str(run_dir), name=f"eval_{subset}", exist_ok=True, device=device, verbose=False)
        row = {"run": run_dir.name, "dataset": dataset_id, "variant": variant, "model": model_name, "seed": seed,
               "subset": subset, "images": n_images, "mAP50": m.box.map50, "mAP50-95": m.box.map}
        for cls_id, name in names.items():
            p, r, ap50, _ = class_metrics(m.box, cls_id)
            row.update({f"{name}_precision": p, f"{name}_recall": r, f"{name}_AP50": ap50})
        row["inference_ms"] = m.speed["inference"]
        rows.append(row)
    return rows, names


def print_table(rows, names):
    w = max(len(r["run"]) for r in rows) + 1
    print(f"{'run':<{w}} {'subset':<11} {'imgs':>4} {'mAP50':>6} {'mAP50-95':>8} {'ms/img':>6}")
    for r in rows:
        print(f"{r['run']:<{w}} {r['subset']:<11} {r['images']:>4} {r['mAP50']:>6.3f} {r['mAP50-95']:>8.3f} "
              f"{r['inference_ms']:>6.1f}")
    print(f"\n{'run':<{w}} {'subset':<11} {'class':<12} {'precision':>9} {'recall':>7} {'AP50':>6}")
    for r in rows:
        for name in names.values():
            print(f"{r['run']:<{w}} {r['subset']:<11} {name:<12} {r[f'{name}_precision']:>9.3f} "
                  f"{r[f'{name}_recall']:>7.3f} {r[f'{name}_AP50']:>6.3f}")
    print("(nan = class has no instances in that subset)")

    # Mean +- std over seeds, per variant/model and subset
    groups = {}
    for r in rows:
        groups.setdefault((r["variant"], r["model"], r["subset"]), []).append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        keys = ["mAP50", "mAP50-95"] + [f"{n}_recall" for n in names.values()]
        print("\nMean +- std over seeds:")
        for (variant, model_name, subset), group in sorted(multi.items()):
            stats = {k: np.array([g[k] for g in group], dtype=float) for k in keys}
            text = "  ".join(f"{k} {np.nanmean(v):.3f}+-{np.nanstd(v):.3f}" for k, v in stats.items())
            print(f"  {variant} {model_name} {subset} ({len(group)} seeds)  {text}")


def evaluate(model_root=MODEL_ROOT, datasets=None, runs=None, device=None, data=None):
    """
    Evaluate trained runs, optionally only those of the dataset IDs in ``datasets`` or the
    runs in ``runs`` ('<dataset>/<run>' or run name). Writes model/<dataset>/comparison.csv.
    """
    found = dataset.find_runs(model_root)
    if datasets:
        found = [(r, d) for r, d in found if d in datasets]
    if runs:
        found = [(r, d) for r, d in found if r.name in runs or f"{d}/{r.name}" in runs]
    if not found:
        print(f"No trained runs found in {model_root} (expected <dataset>/<run>/weights/best.pt); "
              f"available: {[f'{d}/{r.name}' for r, d in dataset.find_runs(model_root)]}")
        return []

    by_dataset = {}
    for run_dir, dataset_id in found:
        print(f"Evaluating {dataset_id}/{run_dir.name} ...")
        run_rows, run_names = evaluate_run(run_dir, dataset_id, data, device)
        group = by_dataset.setdefault(dataset_id, {"rows": [], "names": {}})
        group["rows"] += run_rows
        group["names"].update(run_names)

    all_rows = []
    for dataset_id, group in by_dataset.items():
        rows, names = group["rows"], group["names"]
        if not rows:
            continue
        columns = ["run", "dataset", "variant", "model", "seed", "subset", "images", "mAP50", "mAP50-95"]
        columns += [f"{n}_{m}" for n in names.values() for m in ("precision", "recall", "AP50")] + ["inference_ms"]
        out = Path(model_root) / dataset_id / "comparison.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
        print(f"\n=== dataset {dataset_id}")
        print_table(rows, names)
        if not any(r["subset"] == "test_white" for r in rows):
            print("NOTE: no test_white split; the white-background comparison (primary metric) is not possible yet.")
        print(f"Wrote {paths.to_root_relative(out)}; plots in model/{dataset_id}/<run>/eval_<subset>/")
        all_rows += rows
    return all_rows
