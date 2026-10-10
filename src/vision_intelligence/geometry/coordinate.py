import numpy as np


def depth_at(depth_mm, u, v, window=5):
    """
    Robust depth (mm) at pixel (u, v): median of the valid (non-zero) depths in a
    ``window x window`` patch around the pixel. Returns None if the patch has no
    valid depth, which happens on reflective/dark surfaces and object edges.
    """
    h, w = depth_mm.shape
    r = window // 2
    patch = depth_mm[max(v - r, 0) : min(v + r + 1, h), max(u - r, 0) : min(u + r + 1, w)]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject_pixel(u, v, z_mm, intrinsics):
    """
    Pinhole back-projection of pixel (u, v) with depth ``z_mm`` into a 3D point
    (X, Y, Z) in mm in the color camera frame: X right, Y down, Z forward.
    Lens distortion is ignored, which is fine for a first check.
    """
    x = (u - intrinsics.cx) * z_mm / intrinsics.fx
    y = (v - intrinsics.cy) * z_mm / intrinsics.fy
    return x, y, z_mm
