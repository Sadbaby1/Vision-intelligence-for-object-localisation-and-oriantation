"""
Evaluate trained models on the held-out splits (METHODOLOGY.md Phase 4).

For every run in ``model/<variant>_s<seed>/`` the best weights are scored on the ``test``
split of the dataset the run was trained on (read from the run's ``args.yaml``, or
``--data``), and on ``<dataset>_white.yaml`` if it exists (filters on white backgrounds).
Image size is the one used for training. Results go to:

    model/<run>/eval_<subset>/      Ultralytics plots (confusion matrix, PR curves, predictions)
    model/comparison.csv            one row per run and subset, with metrics for every class

Filter recall on ``test_white`` is the primary metric: a missed filter lowers recall.
"""

import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from vision_intelligence import paths
from vision_intelligence.detection import dataset
from vision_intelligence.detection.train import COMMON, DATASETS_ROOT, MODEL_ROOT, write_dataset_yaml


def find_runs(model_root=MODEL_ROOT):
    """Run folders ``<variant>_s<seed>`` that contain ``weights/best.pt``."""
    runs = []
    for run_dir in sorted(Path(model_root).glob("*_s*")):
        variant, _, seed = run_dir.name.rpartition("_s")
        if (run_dir / "weights" / "best.pt").exists() and seed.isdigit():
            runs.append((run_dir, variant, int(seed)))
    return runs


def class_metrics(box, cls_id):
    """(precision, recall, AP50, AP50-95) of one class, or NaNs if it has no instances in the subset."""
    present = [int(c) for c in box.ap_class_index]
    if cls_id not in present:
        return (float("nan"),) * 4
    return tuple(float(v) for v in box.class_result(present.index(cls_id)))


def run_dataset(run_dir, variant, data=None, datasets_root=DATASETS_ROOT):
    """Dataset YAML for a run: --data, else the one recorded in args.yaml, else this repo's layout."""
    if data:
        return Path(data).resolve()
    recorded = dataset.run_args(run_dir / "weights" / "best.pt").get("data")
    if recorded and Path(recorded).exists():
        return Path(recorded).resolve()
    return write_dataset_yaml(variant, datasets_root)


def evaluate_run(run_dir, variant, seed, data=None, datasets_root=DATASETS_ROOT, device=None):
    data_yaml = run_dataset(run_dir, variant, data, datasets_root)
    subsets = {"test": data_yaml}
    white_yaml = data_yaml.with_name(f"{data_yaml.stem}_white.yaml")
    if white_yaml.exists():
        subsets["test_white"] = white_yaml

    args = dataset.run_args(run_dir / "weights" / "best.pt")
    imgsz = int(args.get("imgsz", COMMON["imgsz"]))
    model = YOLO(str(run_dir / "weights" / "best.pt"))
    names = {int(k): v for k, v in model.names.items()}

    rows = []
    for subset, yaml_path in subsets.items():
        folder = dataset.split_dir(yaml_path, "test")
        n_images = len(dataset.list_images(folder)) if folder else 0
        if n_images == 0:
            print(f"WARNING: no test images for {run_dir.name}/{subset} ({folder}), skipping")
            continue
        print(f"  {subset}: {n_images} images from {folder}, imgsz {imgsz}")
        m = model.val(data=str(yaml_path), split="test", imgsz=imgsz, batch=COMMON["batch"],
                      project=str(run_dir), name=f"eval_{subset}", exist_ok=True, device=device, verbose=False)
        row = {"run": run_dir.name, "variant": variant, "seed": seed, "subset": subset, "images": n_images,
               "mAP50": m.box.map50, "mAP50-95": m.box.map}
        for cls_id, name in names.items():
            p, r, ap50, _ = class_metrics(m.box, cls_id)
            row.update({f"{name}_precision": p, f"{name}_recall": r, f"{name}_AP50": ap50})
        row["inference_ms"] = m.speed["inference"]
        rows.append(row)
    return rows, names


def print_table(rows, names):
    print(f"{'run':<10} {'subset':<11} {'imgs':>4} {'mAP50':>6} {'mAP50-95':>8} {'ms/img':>6}")
    for r in rows:
        print(f"{r['run']:<10} {r['subset']:<11} {r['images']:>4} {r['mAP50']:>6.3f} {r['mAP50-95']:>8.3f} "
              f"{r['inference_ms']:>6.1f}")
    print(f"\n{'run':<10} {'subset':<11} {'class':<12} {'precision':>9} {'recall':>7} {'AP50':>6}")
    for r in rows:
        for name in names.values():
            print(f"{r['run']:<10} {r['subset']:<11} {name:<12} {r[f'{name}_precision']:>9.3f} "
                  f"{r[f'{name}_recall']:>7.3f} {r[f'{name}_AP50']:>6.3f}")
    print("(nan = class has no instances in that subset)")

    # Mean +- std over seeds, per variant and subset
    groups = {}
    for r in rows:
        groups.setdefault((r["variant"], r["subset"]), []).append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        keys = ["mAP50", "mAP50-95"] + [f"{n}_recall" for n in names.values()]
        print("\nMean +- std over seeds:")
        for (variant, subset), group in sorted(multi.items()):
            stats = {k: np.array([g[k] for g in group], dtype=float) for k in keys}
            text = "  ".join(f"{k} {np.nanmean(v):.3f}+-{np.nanstd(v):.3f}" for k, v in stats.items())
            print(f"  {variant:<5} {subset:<11} ({len(group)} seeds)  {text}")


def evaluate(model_root=MODEL_ROOT, datasets_root=DATASETS_ROOT, runs=None, device=None, data=None):
    """Evaluate all runs (or only the run names in ``runs``) and write ``model/comparison.csv``."""
    found = find_runs(model_root)
    if runs:
        found = [r for r in found if r[0].name in runs]
    if not found:
        print(f"No trained runs found in {model_root} (expected <variant>_s<seed>/weights/best.pt)")
        return []

    rows, names = [], {}
    for run_dir, variant, seed in found:
        print(f"Evaluating {run_dir.name} ...")
        run_rows, run_names = evaluate_run(run_dir, variant, seed, data, datasets_root, device)
        rows += run_rows
        names.update(run_names)
    if not rows:
        return []

    columns = ["run", "variant", "seed", "subset", "images", "mAP50", "mAP50-95"]
    columns += [f"{n}_{m}" for n in names.values() for m in ("precision", "recall", "AP50")] + ["inference_ms"]
    out = Path(model_root) / "comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    print()
    print_table(rows, names)
    if not any(r["subset"] == "test_white" for r in rows):
        print("\nNOTE: no test_white split; the white-background comparison (primary metric) is not possible yet.")
    print(f"\nWrote {paths.to_root_relative(out)}; plots in model/<run>/eval_<subset>/")
    return rows
