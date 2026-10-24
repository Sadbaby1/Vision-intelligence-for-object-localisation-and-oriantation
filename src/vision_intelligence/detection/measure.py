"""
Repeatability test (METHODOLOGY.md Phase 5): capture a static scene N times with the camera,
localise every part in every frame, and report per part the mean and spread of position,
yaw and size. Parts are followed across frames by class and image position.

    outputs/measure/<timestamp>_parts.csv   one row per part: mean +- std of every value
    outputs/measure/<timestamp>_frames.csv  every detection of every frame
    outputs/measure/<timestamp>.png         last frame with part numbers
"""

import csv
import time

import cv2 as cv
import numpy as np

from vision_intelligence import paths
from vision_intelligence.detection.live import PartDetector, draw

OUT_DIR = paths.OUTPUTS / "measure"
MATCH_PX = 40  # a detection belongs to a part if its centre is within this distance of the part's first centre
VALUES = [("x_mm", lambda d: d["xyz_mm"][0]), ("y_mm", lambda d: d["xyz_mm"][1]), ("z_mm", lambda d: d["xyz_mm"][2]),
          ("length_mm", lambda d: d["size_mm"][0]), ("width_mm", lambda d: d["size_mm"][1]),
          ("board_x_mm", lambda d: d["xyz_board_mm"][0]), ("board_y_mm", lambda d: d["xyz_board_mm"][1]),
          ("height_mm", lambda d: d["height_mm"]), ("height_charuco_mm", lambda d: d["height_charuco_mm"])]


def circular_stats_180(degrees):
    """Mean and spread of angles with 180-degree symmetry (e.g. -89 and +89 are 2 degrees apart)."""
    a = np.radians(np.asarray(degrees, float)) * 2
    c, s = np.cos(a).mean(), np.sin(a).mean()
    mean = np.degrees(np.arctan2(s, c)) / 2
    r = min(np.hypot(c, s), 1.0)
    std = np.degrees(np.sqrt(-2 * np.log(r))) / 2 if r > 0 else 90.0
    return float((mean + 90) % 180 - 90), float(std)


def track(frames):
    """Group detections of all frames into parts (same class, centre within MATCH_PX of the part's first centre)."""
    parts = []
    for i, dets in enumerate(frames):
        for d in dets:
            best = min((p for p in parts if p["class"] == d["class"] and i not in p["frames"]),
                       key=lambda p: np.hypot(*np.subtract(p["center"], d["center_px"])), default=None)
            if best is not None and np.hypot(*np.subtract(best["center"], d["center_px"])) <= MATCH_PX:
                best["dets"].append(d)
                best["frames"].add(i)
            else:
                parts.append({"class": d["class"], "center": d["center_px"], "dets": [d], "frames": {i}})
    return parts


def summarise(part, n_frames):
    dets = [d for d in part["dets"] if d.get("xyz_mm")]
    row = {"class": part["class"], "seen_in_frames": f"{len(part['dets'])}/{n_frames}",
           "u_px": part["center"][0], "v_px": part["center"][1]}
    for name, get in VALUES:
        try:
            vals = np.array([get(d) for d in dets], float)
        except (KeyError, TypeError):
            continue
        if vals.size:
            row[f"{name}_mean"], row[f"{name}_std"] = round(float(vals.mean()), 1), round(float(vals.std()), 2)
    for key in ("yaw_deg", "yaw_board_deg"):
        angles = [d[key] for d in part["dets"] if d.get(key) is not None]
        if angles:
            mean, std = circular_stats_180(angles)
            row[f"{key}_mean"], row[f"{key}_std"] = round(mean, 1), round(std, 2)
    row["size_ok_share"] = round(float(np.mean([bool(d.get("size_ok")) for d in dets])), 2) if dets else None
    row["methods"] = ",".join(sorted({d["method"] for d in part["dets"]}))
    return row


def measure(weights=None, run_name=None, n_frames=10, conf=0.5, imgsz=None, device=None, stretch=False):
    from vision_intelligence.camera import Camera

    detector = PartDetector(weights, run_name, conf, imgsz, device, stretch)
    frames, camera, last_color = [], None, None
    try:
        camera = Camera()
        camera.start()
        for _ in range(5):  # let exposure settle
            camera.get_frames()
        print(f"Capturing {n_frames} frames of a static scene - keep the scene and camera still ...")
        while len(frames) < n_frames:
            result = camera.get_frames()
            if result is None:
                continue
            color, depth_mm = result
            frames.append(detector(color, depth_mm, camera.intrinsics))
            last_color = color
            time.sleep(0.2)
    finally:
        if camera is not None:
            camera.release()

    parts = sorted(track(frames), key=lambda p: (p["center"][1], p["center"][0]))
    rows = [summarise(p, n_frames) for p in parts]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT_DIR / f"{stamp}_parts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["part"] + columns, restval="")
        writer.writeheader()
        writer.writerows({"part": i + 1, **r} for i, r in enumerate(rows))
    with (OUT_DIR / f"{stamp}_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "class", "confidence", "method", "x_mm", "y_mm", "z_mm", "yaw_deg", "size_ok"])
        for i, dets in enumerate(frames):
            for d in dets:
                writer.writerow([i, d["class"], round(d["confidence"], 3), d["method"],
                                 *(d["xyz_mm"] or (None, None, None)), round(d["yaw_deg"], 1), d.get("size_ok")])
    image = draw(last_color.copy(), frames[-1], detector.variant)
    for i, p in enumerate(parts):
        cv.putText(image, f"#{i + 1}", p["center"], cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv.putText(image, f"#{i + 1}", p["center"], cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv.imwrite(str(OUT_DIR / f"{stamp}.png"), image)

    print(f"\n{'#':>2} {'class':<11} {'seen':>5} {'X mean±std':>15} {'Y mean±std':>15} {'Z mean±std':>15} "
          f"{'yaw mean±std':>14} {'height':>12}")
    for i, r in enumerate(rows, 1):
        def ms(k):
            return f"{r[k + '_mean']:.1f}±{r[k + '_std']:.1f}" if k + "_mean" in r else "-"
        print(f"{i:>2} {r['class']:<11} {r['seen_in_frames']:>5} {ms('x_mm'):>15} {ms('y_mm'):>15} {ms('z_mm'):>15} "
              f"{ms('yaw_board_deg') if 'yaw_board_deg_mean' in r else ms('yaw_deg'):>14} {ms('height_mm'):>12}")
    print(f"\nMETHODOLOGY pass criterion: std < 3 mm. Results: {paths.to_root_relative(OUT_DIR)}/{stamp}_*.csv, {stamp}.png")
    return rows
