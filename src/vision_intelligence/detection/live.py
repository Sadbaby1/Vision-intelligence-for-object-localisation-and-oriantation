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
from vision_intelligence.geometry.board import load_depth_plane, load_pose
from vision_intelligence.geometry.depth import make_rgbd_input
from vision_intelligence.geometry.localize import load_parts, localize_result
from vision_intelligence.visualization import put_text_lines

WINDOW = "Vision Intelligence - live detection"
SNAPSHOT_DIR = paths.OUTPUTS / "live"
MAX_DISPLAY_WIDTH = 1280
# Box colours (BGR), one set per model type so RGB and RGB-D recordings are easy to tell apart.
# Green and red are left free for ground truth and missed parts in `test`.
COLORS = {
    "rgb": {"filter": (0, 165, 255), "frame": (255, 0, 255), "Air_Filter": (255, 255, 0)},   # orange, magenta, cyan
    "rgbd": {"filter": (0, 255, 255), "frame": (255, 120, 0), "Air_Filter": (255, 90, 200)},  # yellow, blue, violet
}
FALLBACK_COLORS = [(255, 255, 255), (128, 128, 128), (0, 128, 128), (128, 128, 0)]


def class_color(name, variant="rgb"):
    return COLORS[variant].get(name) or FALLBACK_COLORS[sum(map(ord, name)) % len(FALLBACK_COLORS)]


class PartDetector:
    """A trained model plus 3D localisation; call it with (colour, depth_mm, intrinsics)."""

    def __init__(self, weights=None, run_name=None, conf=0.5, imgsz=None, device=None, stretch=False, board=True):
        self.weights = dataset.resolve_weights(run_name, weights)
        info = dataset.run_info(self.weights)
        self.imgsz = imgsz or int(dataset.run_args(self.weights).get("imgsz", 640))
        self.conf, self.device, self.stretch = conf, device, stretch
        self.model = YOLO(str(self.weights))
        self.channels = int(self.model.model.model[0].conv.weight.shape[1])
        self.variant = "rgbd" if self.channels == 4 else "rgb"  # also selects the box colours
        self.rgbd_config = info.get("rgbd_config")
        if self.channels == 4 and not self.rgbd_config:
            raise ValueError(f"{self.weights} is an RGB-D model but its run_info.json has no rgbd_config "
                             "(local-height settings); the 4th channel cannot be rebuilt")
        self.parts = load_parts()
        self.board_pose = load_pose() if board else None
        self.depth_plane = load_depth_plane() if board else None  # board plane measured by the depth sensor
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
        return localize_result(pred, depth_mm, intrinsics, self.parts, self.board_pose, scale, self.depth_plane)


LABEL_MODES = ("full", "short", "off")  # cycled with the 'l' key in `detect`


def label_lines(d, mode="full"):
    """Text shown at a detection's centre: 'full' (all values), 'short' (class + height or Z), 'off'."""
    if mode == "off":
        return []
    if mode == "short":
        if d.get("height_mm") is not None:
            return [f"{d['class']} h{d['height_mm']:.0f}mm"]
        return [f"{d['class']} Z{d['xyz_mm'][2]:.0f}mm" if d.get("xyz_mm") else d["class"]]
    first = f"{d['class']} {d['confidence']:.2f}  yaw {d.get('yaw_board_deg', d['yaw_deg']):.0f}deg"
    lines = [first]
    if d.get("xyz_mm"):
        x, y, z = d["xyz_mm"]
        lines.append(f"cam X{x:.0f} Y{y:.0f} Z{z:.0f}mm")
        if d.get("xyz_board_mm"):
            bx, by, _ = d["xyz_board_mm"]
            lines.append(f"board X{bx:.0f} Y{by:.0f} h{d['height_mm']:.0f}mm")
        flags = []  # warnings go on the first line so a label has at most 3 lines
        if not d.get("size_ok", True):
            flags.append(f"size {d['size_mm'][0]:.0f}x{d['size_mm'][1]:.0f}?")
        if d.get("frame_ok") is False:
            flags.append("walls?")
        if flags:
            lines[0] = f"{first}  {' '.join(flags)}"
    else:
        lines.append("no depth")
    return lines


def draw(image, dets, variant="rgb", label_mode="full"):
    """
    Draw each box in the colour of its class for the model type (``variant`` 'rgb' or 'rgbd'),
    a dot at its centre, and the label as white text on a black box centred on that centre
    point (just below the dot), with a thin border in the class colour. Sizes scale with the
    image (reference: 1280 px wide).
    """
    s = max(image.shape[1] / 1280, 0.75)
    font, thick, line_h, pad = 0.45 * s, 1, int(17 * s), int(4 * s)
    face = cv.FONT_HERSHEY_SIMPLEX
    h_img, w_img = image.shape[:2]
    for d in dets:
        color = class_color(d["class"], variant)
        pts = np.asarray(d["corners"]).astype(np.int32)
        cv.polylines(image, [pts], True, color, max(int(round(3 * s)), 1))
        cx, cy = d["center_px"]
        dot = max(int(round(5 * s)), 2)
        cv.circle(image, (cx, cy), dot, color, -1)
        cv.circle(image, (cx, cy), dot, (0, 0, 0), 1)

        lines = label_lines(d, label_mode)
        if not lines:
            continue
        widths = [cv.getTextSize(t, face, font, thick)[0][0] for t in lines]
        box_w, box_h = max(widths) + 2 * pad, len(lines) * line_h + pad
        x0 = int(np.clip(cx - box_w // 2, 0, max(w_img - box_w, 0)))
        y0 = int(np.clip(cy + dot + 3, 0, max(h_img - box_h, 0)))
        cv.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), color, 1)
        for i, text in enumerate(lines):
            org = (x0 + pad, y0 + (i + 1) * line_h - pad // 2)
            cv.putText(image, text, org, face, font, (255, 255, 255), thick, cv.LINE_AA)
    return image


def run(weights=None, conf=0.5, imgsz=None, device=None, stretch=False, run_name=None, board=True):
    """Live view. Model: ``weights``, else ``run_name`` ('<dataset>/<run>'), else the latest trained run."""
    from vision_intelligence.camera import Camera

    detector = PartDetector(weights, run_name, conf, imgsz, device, stretch, board)
    orbbec = None
    try:
        orbbec = Camera()
        orbbec.start()
        print("Live detection. 'l' = labels full/short/off, 's' = save snapshot, 'q' = quit.")
        last, fps = time.monotonic(), 0.0
        label_mode = 0
        while True:
            result = orbbec.get_frames()
            if result is None:
                continue
            color, depth_mm = result
            dets = detector(color, depth_mm, orbbec.intrinsics)
            annotated = draw(color.copy(), dets, detector.variant, LABEL_MODES[label_mode])

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            counts = "   ".join(f"{n}: {sum(d['class'] == n for d in dets)}" for n in detector.model.names.values())
            model_type = "RGB-D model" if detector.variant == "rgbd" else "RGB model"
            put_text_lines(annotated, [f"{model_type}   {counts}   conf >= {conf}   {fps:.1f} FPS"])

            scale = min(1.0, MAX_DISPLAY_WIDTH / annotated.shape[1])
            display = cv.resize(annotated, None, fx=scale, fy=scale) if scale < 1.0 else annotated
            cv.imshow(WINDOW, display)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                label_mode = (label_mode + 1) % len(LABEL_MODES)
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
