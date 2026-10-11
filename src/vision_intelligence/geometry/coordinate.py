import numpy as np


def depth_at(depth_mm, u, v, window=5):
    """
    Robust depth (mm) at pixel (u, v): median of the valid (non-zero) depths in a
    ``window x window`` patch around the pixel. Returns None if the patch has no
    valid depth, which happens on reflective/dark surfaces and object edges.
    """
    h, w = depth_mm.shape
    r = window // 2
    patch = depth_mm[
        max(v - r, 0) : min(v + r + 1, h), max(u - r, 0) : min(u + r + 1, w)
    ]
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


class CoordinateTransformer:
    """
    Converts an object's pixel position into robot work-object coordinates,
    using the ArUco marker as the known reference point.
    """

    def __init__(self, marker_world_x, marker_world_y, pixel_mm_ratio):
        # Real-world robot coordinates of the marker's center (from calibration jog)
        self.marker_world_x = marker_world_x
        self.marker_world_y = marker_world_y

        # Pixel-to-mm ratio from ArucoCalibrator
        self.pixel_mm_ratio = pixel_mm_ratio

    def transform(self, object_pixel_x, object_pixel_y, marker_pixel_x, marker_pixel_y):
        """
        Converts an object's pixel position into robot coordinates (mm),
        relative to the marker's known real-world position.
        """
        # Offset in pixels between the object and the marker
        offset_x_px = object_pixel_x - marker_pixel_x
        offset_y_px = object_pixel_y - marker_pixel_y

        # Convert pixel offset to millimeters using the calibration ratio
        offset_x_mm = offset_x_px / self.pixel_mm_ratio
        offset_y_mm = offset_y_px / self.pixel_mm_ratio

        # Add offset to the marker's known robot-world position
        robot_x = self.marker_world_x + offset_x_mm
        robot_y = self.marker_world_y + offset_y_mm

        return robot_x, robot_y
