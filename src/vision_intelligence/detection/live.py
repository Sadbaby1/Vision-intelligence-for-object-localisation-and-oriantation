"""
Live detection test: run the trained YOLO-OBB model on the Orbbec colour stream and
show, per detection, class, confidence, approximate 3D centre (camera frame, mm) and yaw.

The 3D position is a first approximation: depth is the median in a small window at
the box centre. For a hollow frame that is the surface below the frame, not the frame
itself (proper part depth is Phase 5 of METHODOLOGY.md).
"""

import time
from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from vision_intelligence import paths
from vision_intelligence.camera import Camera
from vision_intelligence.geometry.coordinate import deproject_pixel, depth_at
from vision_intelligence.visualization import put_text_lines

WINDOW = "Vision Intelligence - live detection"
DEFAULT_WEIGHTS = paths.MODEL / "rgb_s0" / "weights" / "best.pt"
SNAPSHOT_DIR = paths.OUTPUTS / "live"
MAX_DISPLAY_WIDTH = 1280
COLORS = {"filter": (0, 165, 255), "frame": (255, 0, 255)}  # BGR: orange, magenta


def yaw_deg(corners):
    """
    Angle (deg) of the box's long side against the image x axis, folded to [-90, 90)
    because parts are 180-degree symmetric. Image y points down, so positive = clockwise.
    """
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    dx, dy = max(edges, key=np.linalg.norm)
    return float((np.degrees(np.arctan2(dy, dx)) + 90) % 180 - 90)


def detections(result, depth_mm, intrinsics):
    """Turn one Ultralytics OBB result into a list of dicts with class, confidence, 3D centre and yaw."""
    out = []
    obb = result.obb
    if obb is None:
        return out
    for corners, cls_id, conf in zip(obb.xyxyxyxy.cpu().numpy(), obb.cls.cpu().numpy(), obb.conf.cpu().numpy()):
        u, v = (int(round(c)) for c in corners.mean(axis=0))
        z = depth_at(depth_mm, u, v, window=9)
        xyz = deproject_pixel(u, v, z, intrinsics) if z is not None else None
        out.append({
            "class": result.names[int(cls_id)],
            "confidence": float(conf),
            "corners": corners,
            "center_px": (u, v),
            "xyz_mm": xyz,
            "yaw_deg": yaw_deg(corners),
        })
    return out


def draw(image, dets):
    for d in dets:
        color = COLORS.get(d["class"], (0, 255, 0))
        pts = d["corners"].astype(np.int32)
        cv.polylines(image, [pts], True, color, 3)
        cv.circle(image, d["center_px"], 5, color, -1)
        xyz = d["xyz_mm"]
        pos = f"X{xyz[0]:.0f} Y{xyz[1]:.0f} Z{xyz[2]:.0f}mm" if xyz else "no depth"
        lines = [f"{d['class']} {d['confidence']:.2f}  yaw {d['yaw_deg']:.0f}deg", pos]
        x, y = int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 35, 20)
        for i, text in enumerate(lines):
            org = (x, y + i * 22)
            cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def run(weights=DEFAULT_WEIGHTS, conf=0.5, imgsz=1280, device=None):
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found; train a model first (python -m vision_intelligence train)")
    model = YOLO(str(weights))
    print(f"Loaded {weights}")

    orbbec = None
    try:
        orbbec = Camera()
        orbbec.start()
        print("Live detection. 's' = save snapshot, 'q' = quit.")
        last = time.monotonic()
        fps = 0.0
        while True:
            result = orbbec.get_frames()
            if result is None:
                continue
            color, depth_mm = result

            pred = model.predict(color, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            dets = detections(pred, depth_mm, orbbec.intrinsics)
            annotated = draw(color.copy(), dets)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            counts = {c: sum(d["class"] == c for d in dets) for c in pred.names.values()}
            summary = "   ".join(f"{name}: {n}" for name, n in counts.items())
            put_text_lines(annotated, [f"{summary}   conf >= {conf}   {fps:.1f} FPS"])

            scale = min(1.0, MAX_DISPLAY_WIDTH / annotated.shape[1])
            display = cv.resize(annotated, None, fx=scale, fy=scale) if scale < 1.0 else annotated
            cv.imshow(WINDOW, display)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv.imwrite(str(SNAPSHOT_DIR / f"{stamp}_color.png"), color)
                cv.imwrite(str(SNAPSHOT_DIR / f"{stamp}_detections.png"), annotated)
                print(f"Saved snapshot {stamp}: " + "; ".join(
                    f"{d['class']} {d['confidence']:.2f} yaw {d['yaw_deg']:.0f}" for d in dets))
    finally:
        if orbbec is not None:
            orbbec.release()
        cv.destroyAllWindows()
