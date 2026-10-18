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
SNAPSHOT_DIR = paths.OUTPUTS / "live"
MAX_DISPLAY_WIDTH = 1280
COLORS = {"filter": (0, 165, 255), "frame": (255, 0, 255), "Air_Filter": (255, 255, 0)}  # BGR: orange, magenta, cyan
FALLBACK_COLORS = [(0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255)]


def class_color(name):
    return COLORS.get(name) or FALLBACK_COLORS[sum(map(ord, name)) % len(FALLBACK_COLORS)]


def yaw_deg(corners):
    """
    Angle (deg) of the box's long side against the image x axis, folded to [-90, 90)
    because parts are 180-degree symmetric. Image y points down, so positive = clockwise.
    """
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    dx, dy = max(edges, key=np.linalg.norm)
    return float((np.degrees(np.arctan2(dy, dx)) + 90) % 180 - 90)


def detections(result, depth_mm=None, intrinsics=None, scale=(1.0, 1.0)):
    """
    Turn one Ultralytics OBB result into a list of dicts with class, confidence, 3D centre
    and yaw. Without depth/intrinsics the 3D centre is None. ``scale`` (sx, sy) maps box
    corners back to the original image when the model saw a resized copy.
    """
    out = []
    obb = result.obb
    if obb is None:
        return out
    for corners, cls_id, conf in zip(obb.xyxyxyxy.cpu().numpy(), obb.cls.cpu().numpy(), obb.conf.cpu().numpy()):
        corners = corners * np.asarray(scale, dtype=np.float32)
        u, v = (int(round(c)) for c in corners.mean(axis=0))
        z = depth_at(depth_mm, u, v, window=9) if depth_mm is not None and intrinsics is not None else None
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
    """Draw boxes and labels; text and line sizes scale with the image (0.6 at 1280 px wide)."""
    s = max(image.shape[1] / 1280, 0.75)
    font, thick, line_h = 0.6 * s, max(int(round(2 * s)), 1), int(22 * s)
    for d in dets:
        color = class_color(d["class"])
        pts = d["corners"].astype(np.int32)
        cv.polylines(image, [pts], True, color, max(int(round(3 * s)), 1))
        cv.circle(image, d["center_px"], max(int(round(5 * s)), 2), color, -1)
        lines = [f"{d['class']} {d['confidence']:.2f}  yaw {d['yaw_deg']:.0f}deg"]
        xyz = d["xyz_mm"]
        if xyz:
            lines.append(f"X{xyz[0]:.0f} Y{xyz[1]:.0f} Z{xyz[2]:.0f}mm")
        x, y = int(pts[:, 0].min()), max(int(pts[:, 1].min()) - line_h * len(lines) + line_h // 2, line_h)
        for i, text in enumerate(lines):
            org = (x, y + i * line_h)
            cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick + 2)
            cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, font, color, thick)
    return image


def run(weights=None, conf=0.5, imgsz=None, device=None, stretch=False, run_name=None):
    """
    Model: ``weights``, else ``run_name`` ('<dataset>/<run>'), else the latest trained run.
    ``imgsz`` defaults to the model's training image size (from the run's args.yaml).
    ``stretch``: resize each camera frame to imgsz x imgsz without keeping the aspect ratio
    before detection. Use it only if the training images were stretched the same way
    (e.g. Roboflow "Resize: Stretch to 640x640"); otherwise leave it off (letterboxing).
    """
    from vision_intelligence.detection.dataset import resolve_weights, run_args

    weights = resolve_weights(run_name, weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found; train a model first (python -m vision_intelligence train)")
    imgsz = imgsz or int(run_args(weights).get("imgsz", 640))
    model = YOLO(str(weights))
    print(f"Loaded {weights} (classes {model.names}), imgsz {imgsz}, {'stretch' if stretch else 'letterbox'}")

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

            if stretch:
                h, w = color.shape[:2]
                model_input, scale = cv.resize(color, (imgsz, imgsz)), (w / imgsz, h / imgsz)
            else:
                model_input, scale = color, (1.0, 1.0)
            pred = model.predict(model_input, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            dets = detections(pred, depth_mm, orbbec.intrinsics, scale)
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
