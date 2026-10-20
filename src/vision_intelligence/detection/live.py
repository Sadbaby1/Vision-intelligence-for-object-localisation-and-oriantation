"""
Live detection with 3D localisation on the Orbbec camera.

``PartDetector`` = trained model + localisation (geometry/localize.py). It works for RGB
models and for RGB-D models, for which the 4th input channel (local height) is built from the
live depth with exactly the settings stored with the model (run_info.json -> rgbd_config).
Per detection it gives class, confidence, 3D position of the top-face centre (camera frame,
and board frame if config/board_pose.json exists), yaw and a size check.
"""

import time

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from vision_intelligence import paths
from vision_intelligence.detection import dataset
from vision_intelligence.geometry.board import load_pose
from vision_intelligence.geometry.depth import make_rgbd_input
from vision_intelligence.geometry.localize import load_parts, localize_result
from vision_intelligence.visualization import put_text_lines

WINDOW = "Vision Intelligence - live detection"
SNAPSHOT_DIR = paths.OUTPUTS / "live"
MAX_DISPLAY_WIDTH = 1280
COLORS = {"filter": (0, 165, 255), "frame": (255, 0, 255), "Air_Filter": (255, 255, 0)}  # BGR: orange, magenta, cyan
FALLBACK_COLORS = [(0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255)]


def class_color(name):
    return COLORS.get(name) or FALLBACK_COLORS[sum(map(ord, name)) % len(FALLBACK_COLORS)]


class PartDetector:
    """A trained model plus 3D localisation; call it with (colour, depth_mm, intrinsics)."""

    def __init__(self, weights=None, run_name=None, conf=0.5, imgsz=None, device=None, stretch=False, board=True):
        self.weights = dataset.resolve_weights(run_name, weights)
        info = dataset.run_info(self.weights)
        self.imgsz = imgsz or int(dataset.run_args(self.weights).get("imgsz", 640))
        self.conf, self.device, self.stretch = conf, device, stretch
        self.model = YOLO(str(self.weights))
        self.channels = int(self.model.model.model[0].conv.weight.shape[1])
        self.rgbd_config = info.get("rgbd_config")
        if self.channels == 4 and not self.rgbd_config:
            raise ValueError(f"{self.weights} is an RGB-D model but its run_info.json has no rgbd_config "
                             "(local-height settings); the 4th channel cannot be rebuilt")
        self.parts = load_parts()
        self.board_pose = load_pose() if board else None
        print(f"Model {paths.to_root_relative(self.weights)}: {'RGB-D' if self.channels == 4 else 'RGB'}, "
              f"classes {self.model.names}, imgsz {self.imgsz}, {'stretch' if stretch else 'letterbox'}, "
              f"board frame {'on' if self.board_pose else 'off (no config/board_pose.json)'}")

    def model_input(self, color, depth_mm):
        """3-channel colour, or 4-channel colour + local height for RGB-D models."""
        if self.channels == 4:
            if depth_mm is None:
                raise ValueError("an RGB-D model needs the depth image")
            return make_rgbd_input(color, depth_mm, self.rgbd_config)
        return color

    def __call__(self, color, depth_mm, intrinsics, image=None):
        """
        Detections with localisation. ``image`` is the model input if it is already prepared
        (e.g. a 4-channel test TIFF); otherwise it is built from ``color`` and ``depth_mm``.
        """
        image = self.model_input(color, depth_mm) if image is None else image
        scale = (1.0, 1.0)
        if self.stretch:
            h, w = image.shape[:2]
            image, scale = cv.resize(image, (self.imgsz, self.imgsz)), (w / self.imgsz, h / self.imgsz)
        pred = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        return localize_result(pred, depth_mm, intrinsics, self.parts, self.board_pose, scale)


def label_lines(d):
    """Text shown next to a detection."""
    lines = [f"{d['class']} {d['confidence']:.2f}  yaw {d.get('yaw_board_deg', d['yaw_deg']):.0f}deg"]
    if d.get("xyz_mm"):
        x, y, z = d["xyz_mm"]
        lines.append(f"cam X{x:.0f} Y{y:.0f} Z{z:.0f}mm")
        if d.get("xyz_board_mm"):
            bx, by, _ = d["xyz_board_mm"]
            lines.append(f"board X{bx:.0f} Y{by:.0f} h{d['height_mm']:.0f}mm")
        flags = []
        if not d.get("size_ok", True):
            flags.append(f"size {d['size_mm'][0]:.0f}x{d['size_mm'][1]:.0f}?")
        if d.get("frame_ok") is False:
            flags.append("walls?")
        if flags:
            lines.append(" ".join(flags))
    else:
        lines.append("no depth")
    return lines


def draw(image, dets):
    """Draw boxes and labels; text and line sizes scale with the image (0.6 at 1280 px wide)."""
    s = max(image.shape[1] / 1280, 0.75)
    font, thick, line_h = 0.55 * s, max(int(round(2 * s)), 1), int(20 * s)
    for d in dets:
        color = class_color(d["class"])
        pts = np.asarray(d["corners"]).astype(np.int32)
        cv.polylines(image, [pts], True, color, max(int(round(3 * s)), 1))
        cv.circle(image, d["center_px"], max(int(round(5 * s)), 2), color, -1)
        lines = label_lines(d)
        x, y = int(pts[:, 0].min()), max(int(pts[:, 1].min()) - line_h * len(lines) + line_h // 2, line_h)
        for i, text in enumerate(lines):
            org = (x, y + i * line_h)
            cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick + 2)
            cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, font, color, thick)
    return image


def run(weights=None, conf=0.5, imgsz=None, device=None, stretch=False, run_name=None, board=True):
    """Live view. Model: ``weights``, else ``run_name`` ('<dataset>/<run>'), else the latest trained run."""
    from vision_intelligence.camera import Camera

    detector = PartDetector(weights, run_name, conf, imgsz, device, stretch, board)
    orbbec = None
    try:
        orbbec = Camera()
        orbbec.start()
        print("Live detection. 's' = save snapshot, 'q' = quit.")
        last, fps = time.monotonic(), 0.0
        while True:
            result = orbbec.get_frames()
            if result is None:
                continue
            color, depth_mm = result
            dets = detector(color, depth_mm, orbbec.intrinsics)
            annotated = draw(color.copy(), dets)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            counts = "   ".join(f"{n}: {sum(d['class'] == n for d in dets)}" for n in detector.model.names.values())
            put_text_lines(annotated, [f"{counts}   conf >= {conf}   {fps:.1f} FPS"])

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
                cv.imwrite(str(SNAPSHOT_DIR / f"{stamp}_depth.png"), np.clip(depth_mm, 0, 65535).astype(np.uint16))
                cv.imwrite(str(SNAPSHOT_DIR / f"{stamp}_detections.png"), annotated)
                print(f"Saved snapshot {stamp}: " + "; ".join(
                    f"{d['class']} {d['confidence']:.2f} xyz {d['xyz_mm']} yaw {d['yaw_deg']:.0f}" for d in dets))
    finally:
        if orbbec is not None:
            orbbec.release()
        cv.destroyAllWindows()
