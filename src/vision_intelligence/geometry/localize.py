"""
3D localisation and orientation of detected parts (METHODOLOGY.md Phase 5).

Input per detection: class and rotated box (4 corners, pixels) from the detector, the
aligned depth image (mm) and the colour-camera intrinsics. Output: the 3D position of the
centre of the part's top face, yaw, a size check and quality information.

Depth of the top face (``method``):
    box               solid parts (filter, Air_Filter): robust median depth inside the box,
                      shrunk at the edges so gaps and neighbouring parts do not count
    hole-surface      hollow frames: the surface the frame stands on, seen through its open
                      centre, minus the frame height. The ~8 mm walls are too thin and blurred in
                      depth to be measured directly (measured on the test set: 24 mm instead of 45)
    walls             frames whose centre has no depth: closest wall pixels (biased low)
    outside-surface   frames with neither: surface around the frame minus the frame height
    no-depth          no valid depth at all; only image-based values are given
For frames the measured wall height is reported as a plausibility check (``frame_ok``).

Camera frame: X right, Y down, Z forward from the camera (mm). If a board pose exists
(``config/board_pose.json``, from `calibrate-board`), the position is also given in the
board frame, with height above the board and yaw measured in the table plane.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2 as cv
import numpy as np

from vision_intelligence import paths
from vision_intelligence.geometry.board import camera_to_board

SIZE_TOLERANCE = 0.10        # relative deviation from the part length/width that is still "size ok"
EDGE_MARGIN = 0.08           # box shrink, as a share of the box's narrow side (min 2 px)
MIN_VALID_PIXELS = 30        # fewer valid depth pixels than this -> region unusable
WALL_PERCENTILE = 25         # frames without a visible centre: closest quarter of the wall-ring depths
WALL_TOP_PERCENTILE = 2      # plausibility check: closest 2 % of wall pixels ~ top of the (blurred) walls
MIN_WALL_HEIGHT = 0.4        # measured wall height must be >= this share of the frame height ("frame_ok")


def load_parts(path=paths.PART_DIMENSIONS):
    """{class name: {length, width, height, hollow, wall_mm}} from config/part_dimensions.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def read_intrinsics(path):
    """Intrinsics (fx, fy, cx, cy, width, height) from a session's intrinsics.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = data.get("color", data)
    return SimpleNamespace(**{k: data[k] for k in ("fx", "fy", "cx", "cy", "width", "height")})


# ---------------------------------------------------------------- geometry helpers

def deproject(u, v, z, intrinsics):
    """Pixel (u, v) at depth z (mm) -> camera-frame point (X, Y, Z) in mm."""
    return np.array([(u - intrinsics.cx) * z / intrinsics.fx, (v - intrinsics.cy) * z / intrinsics.fy, z])


def yaw_image_deg(corners):
    """Angle of the box's long side against the image x axis, folded to [-90, 90) (parts are 180-degree symmetric)."""
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    dx, dy = max(edges, key=np.linalg.norm)
    return _fold(np.degrees(np.arctan2(dy, dx)))


def _fold(deg):
    return float((deg + 90) % 180 - 90)


def box_sides_px(corners):
    e1, e2 = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1])
    return max(e1, e2), min(e1, e2)


def polygon_mask(shape, corners):
    mask = np.zeros(shape, np.uint8)
    cv.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    return mask


def shrink(mask, px):
    if px <= 0:
        return mask
    return cv.erode(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))


def inner_corners(corners, inset_px):
    """Corners of the same rotated rectangle shrunk by ``inset_px`` on every side."""
    (cx, cy), (w, h), angle = cv.minAreaRect(corners.astype(np.float32))
    return cv.boxPoints(((cx, cy), (max(w - 2 * inset_px, 1), max(h - 2 * inset_px, 1)), angle))


def robust_median(values):
    """Median after removing outliers (more than 3 robust standard deviations and 5 mm away)."""
    med = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - med))
    keep = np.abs(values - med) <= max(3 * mad, 5.0)
    return float(np.median(values[keep]))


def _valid(depth, mask):
    return depth[(mask > 0) & (depth > 0)].astype(np.float64)


# ---------------------------------------------------------------- depth of one part

def part_depth(depth, corners, part, intrinsics):
    """(z_mm of the top face or None, method, valid fraction of the measured region, surface_mm or None)."""
    long_px, short_px = box_sides_px(corners)
    margin = max(2, round(EDGE_MARGIN * short_px))
    box = polygon_mask(depth.shape, corners)
    region = shrink(box, margin)

    if not part.get("hollow"):
        vals = _valid(depth, region)
        frac = len(vals) / max(int(region.sum()), 1)
        if len(vals) < MIN_VALID_PIXELS:
            return None, "no-depth", frac, None
        return robust_median(vals), "box", frac, None

    # Hollow frame. Its walls (~8 mm) are only a few pixels wide and blurred in the depth image,
    # so their measured height is biased low. The surface the frame stands on, seen through its
    # open centre, is measured reliably: top of frame = that surface - frame height.
    z_guess = np.median(_valid(depth, box)) if _valid(depth, box).size else 1000.0
    wall_px = max(2.0, part.get("wall_mm", 8) * intrinsics.fx / z_guess)
    hole = shrink(polygon_mask(depth.shape, inner_corners(corners, wall_px + 2)), max(2, margin))
    ring = shrink(box, 1) & (1 - polygon_mask(depth.shape, inner_corners(corners, wall_px)))
    hole_vals, ring_vals = _valid(depth, hole), _valid(depth, ring)
    frac = len(hole_vals) / max(int(hole.sum()), 1)

    if hole_vals.size >= MIN_VALID_PIXELS:
        surface = robust_median(hole_vals)
        return surface - part["height"], "hole-surface", frac, surface
    if ring_vals.size >= MIN_VALID_PIXELS:  # centre covered or without depth: closest wall pixels
        return float(np.percentile(ring_vals, WALL_PERCENTILE)), "walls", \
            len(ring_vals) / max(int(ring.sum()), 1), None
    outside = _valid(depth, cv.dilate(box, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) - box)
    if outside.size >= MIN_VALID_PIXELS:
        surface = robust_median(outside)
        return surface - part["height"], "outside-surface", frac, surface
    return None, "no-depth", frac, None


def wall_height(depth, corners, part, intrinsics, surface):
    """Measured height of the frame walls above ``surface`` (closest wall pixels), as a plausibility check."""
    box = polygon_mask(depth.shape, corners)
    z_guess = np.median(_valid(depth, box)) if _valid(depth, box).size else 1000.0
    wall_px = max(2.0, part.get("wall_mm", 8) * intrinsics.fx / z_guess)
    ring_vals = _valid(depth, shrink(box, 1) & (1 - polygon_mask(depth.shape, inner_corners(corners, wall_px))))
    return float(surface - np.percentile(ring_vals, WALL_TOP_PERCENTILE)) if ring_vals.size >= MIN_VALID_PIXELS else None


# ---------------------------------------------------------------- full localisation

def size_check(corners, z, intrinsics, part, tol=SIZE_TOLERANCE):
    """Length and width (mm) of the box at depth z, and whether they match the part (flat or on its side)."""
    pts = [deproject(u, v, z, intrinsics) for u, v in corners]
    sides = [np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])]
    length, width = max(sides), min(sides)
    footprints = [(part["length"], part["width"]), (part["length"], part["height"])]  # flat / on its side
    errors = [(length / L - 1, width / W - 1) for L, W in footprints]
    best = min(errors, key=lambda e: abs(e[0]) + abs(e[1]))
    return round(float(length), 1), round(float(width), 1), all(abs(e) <= tol for e in best), best


def localize(cls, confidence, corners, depth_mm, intrinsics, parts, board_pose=None):
    """
    One detection -> dict with class, confidence, corners, centre_px, yaw, and (when depth is
    usable) xyz_mm, size check and, with a board pose, board-frame position, height and yaw.
    """
    corners = np.asarray(corners, np.float32)
    u, v = (float(c) for c in corners.mean(axis=0))
    out = {"class": cls, "confidence": float(confidence), "corners": corners,
           "center_px": (int(round(u)), int(round(v))), "yaw_deg": yaw_image_deg(corners),
           "xyz_mm": None, "method": "no-depth", "valid_depth": 0.0}
    part = parts.get(cls)
    if depth_mm is None or intrinsics is None or part is None:
        return out

    z, method, frac, surface = part_depth(depth_mm, corners, part, intrinsics)
    out.update(method=method, valid_depth=round(frac, 3))
    if z is None:
        return out
    xyz = deproject(u, v, z, intrinsics)
    length, width, size_ok, size_err = size_check(corners, z, intrinsics, part)
    out.update(xyz_mm=tuple(round(float(c), 1) for c in xyz), size_mm=(length, width), size_ok=size_ok,
               size_error=tuple(round(float(e), 3) for e in size_err))
    if surface is not None and part.get("hollow"):
        measured = wall_height(depth_mm, corners, part, intrinsics, surface)
        out["surface_mm"] = round(float(surface), 1)
        out["wall_height_measured_mm"] = None if measured is None else round(measured, 1)
        out["frame_ok"] = measured is not None and measured >= MIN_WALL_HEIGHT * part["height"]

    if board_pose is not None:
        p_board = camera_to_board(xyz, board_pose)[0]
        # long axis: midpoints of the two short edges, in 3D at the top-face depth
        e01 = np.linalg.norm(corners[1] - corners[0]) >= np.linalg.norm(corners[2] - corners[1])
        a, b = ((corners[1] + corners[2]) / 2, (corners[3] + corners[0]) / 2) if e01 else \
            ((corners[0] + corners[1]) / 2, (corners[2] + corners[3]) / 2)
        ends = camera_to_board(np.array([deproject(*a, z, intrinsics), deproject(*b, z, intrinsics)]), board_pose)
        d = ends[1] - ends[0]
        out.update(xyz_board_mm=tuple(round(float(c), 1) for c in p_board),
                   height_mm=round(float(-p_board[2]), 1),
                   yaw_board_deg=round(_fold(np.degrees(np.arctan2(d[1], d[0]))), 1))
    return out


def localize_result(result, depth_mm, intrinsics, parts, board_pose=None, scale=(1.0, 1.0)):
    """All detections of one Ultralytics OBB result. ``scale`` maps corners back to the original image."""
    obb = result.obb
    if obb is None:
        return []
    names = result.names
    return [
        localize(names[int(c)], conf, corners * np.asarray(scale, np.float32), depth_mm, intrinsics, parts, board_pose)
        for corners, c, conf in zip(obb.xyxyxyxy.cpu().numpy(), obb.cls.cpu().numpy(), obb.conf.cpu().numpy())
    ]
