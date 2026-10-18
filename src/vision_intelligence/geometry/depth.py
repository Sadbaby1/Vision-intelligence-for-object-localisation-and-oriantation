"""
Depth helpers shared by data checks, dataset building, auto-labelling and prediction.
Depth images are uint16/float in millimetres, aligned to the colour image; 0 = no depth.
"""

import cv2 as cv
import numpy as np

KERNEL_PX = 101        # must be wider than the NARROW side of a part, in pixels
MAX_HEIGHT_MM = 100.0  # about 1.5x the tallest part; heights above this become 255


def z(d, u, v, r=3):
    """Median depth (mm) in a (2r+1)x(2r+1) window around pixel (u, v), ignoring zeros. NaN if none."""
    patch = d[max(v - r, 0) : v + r + 1, max(u - r, 0) : u + r + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else float("nan")


def fill_depth_holes(depth_mm, iterations=10):
    """Fill missing (0) depth from neighbours by repeated dilation; leftovers get the image max."""
    d = depth_mm.astype(np.float32)
    invalid = d == 0
    for _ in range(iterations):
        if not invalid.any():
            break
        grown = cv.dilate(d, np.ones((5, 5), np.uint8))
        d[invalid] = grown[invalid]
        invalid = d == 0
    d[invalid] = d.max()
    return d


def local_height_mm(depth_mm, kernel_px=KERNEL_PX, blur_px=0):
    """
    Height (mm) above the local surrounding surface: black-hat transform, closing(d) - d.
    ``blur_px`` > 0 smooths the hole-filled depth with a Gaussian first (sensor noise).
    """
    d = fill_depth_holes(depth_mm)
    if blur_px:
        d = cv.GaussianBlur(d, (blur_px | 1, blur_px | 1), 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv.morphologyEx(d, cv.MORPH_BLACKHAT, kernel)


def local_height(depth_mm, kernel_px=KERNEL_PX, max_height_mm=MAX_HEIGHT_MM, blur_px=0):
    """
    Local height scaled to 0-255 (uint8): the 4th input channel of the RGB-D model.
    Training data and prediction must use the same parameters (stored with each RGB-D dataset).
    """
    height_mm = local_height_mm(depth_mm, kernel_px, blur_px)
    return np.clip(height_mm / max_height_mm * 255, 0, 255).astype(np.uint8)


def make_rgbd_input(bgr, depth_mm, config):
    """
    4-channel RGB-D model input (B, G, R, local height) from a colour image and its aligned
    depth. ``config`` holds kernel_px, max_height_mm and blur_px, exactly as used to build the
    training data (stored in each RGB-D run's run_info.json as ``rgbd_config``).
    """
    height = local_height(depth_mm, config["kernel_px"], config["max_height_mm"], config.get("blur_px", 0))
    return np.dstack([bgr, height])
