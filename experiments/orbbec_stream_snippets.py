"""
import numpy as np
import cv2
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
from camera import camera
pipeline = Pipeline()
config = Config()

# Color stream
color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
config.enable_stream(color_profiles.get_default_video_stream_profile())

# Depth stream
depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
config.enable_stream(depth_profiles.get_default_video_stream_profile())

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is not None:
            data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            if color_frame.get_format() == OBFormat.MJPG:
                color_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif color_frame.get_format() == OBFormat.RGB:
                color_img = cv2.cvtColor(
                    data.reshape(color_frame.get_height(), color_frame.get_width(), 3),
                    cv2.COLOR_RGB2BGR)
            else:
                color_img = None  # handle other formats (YUYV, etc.) as needed
            if color_img is not None:
                cv2.imshow("Color", color_img)

        if depth_frame is not None:
            w, h = depth_frame.get_width(), depth_frame.get_height()
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
            depth_mm = depth.astype(np.float32) * depth_frame.get_depth_scale()
            vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_mm, alpha=0.05), cv2.COLORMAP_JET)
            cv2.imshow("Depth", vis)

        if cv2.waitKey(1) in (27, ord('q')):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()


import numpy as np
import cv2
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

pipeline = Pipeline()
config = Config()

# Color stream
color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
config.enable_stream(color_profiles.get_default_video_stream_profile())

# Depth stream
depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
config.enable_stream(depth_profiles.get_default_video_stream_profile())

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is not None:
            data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            if color_frame.get_format() == OBFormat.MJPG:
                color_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif color_frame.get_format() == OBFormat.RGB:
                color_img = cv2.cvtColor(
                    data.reshape(color_frame.get_height(), color_frame.get_width(), 3),
                    cv2.COLOR_RGB2BGR,
                )
            else:
                color_img = None  # handle other formats (YUYV, etc.) as needed
            if color_img is not None:
                cv2.imshow("Color", color_img)

        if depth_frame is not None:
            w, h = depth_frame.get_width(), depth_frame.get_height()
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
            depth_mm = depth.astype(np.float32) * depth_frame.get_depth_scale()
            vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_mm, alpha=0.05), cv2.COLORMAP_JET
            )
            cv2.imshow("Depth", vis)

        if cv2.waitKey(1) in (27, ord("q")):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
"""

import numpy as np
import cv2 as cv
from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat


class Camera:
    def __init__(self, cam_id=0, serial=None):
        ctx = Context()
        devices = ctx.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("No Orbbec camera found")

        if serial:
            device = devices.get_device_by_serial_number(serial)
        else:
            device = devices.get_device_by_index(cam_id)

        self.pipeline = Pipeline(device)
        config = Config()

        color_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        config.enable_stream(color_profiles.get_default_video_stream_profile())

        depth_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        config.enable_stream(depth_profiles.get_default_video_stream_profile())

        self.pipeline.start(config)
        self.frames = None

    def grab(self, timeout_ms=10000000):
        """Wait for a new set of frames. Returns True if one arrived."""
        self.frames = self.pipeline.wait_for_frames(timeout_ms)
        return self.frames is not None

    def read_color(self):
        """Return the latest color image as BGR, or None."""
        if self.frames is None:
            return None
        color_frame = self.frames.get_color_frame()
        if color_frame is None:
            return None

        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        h, w = color_frame.get_height(), color_frame.get_width()
        fmt = color_frame.get_format()

        if fmt == OBFormat.MJPG:
            return cv.imdecode(data, cv.IMREAD_COLOR)
        if fmt == OBFormat.RGB:
            return cv.cvtColor(data.reshape(h, w, 3), cv.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return data.reshape(h, w, 3)
        if fmt == OBFormat.YUYV:
            return cv.cvtColor(data.reshape(h, w, 2), cv.COLOR_YUV2BGR_YUYV)
        return None

    def read_depth(self, colorize=False):
        """Return depth in millimetres (float32), or a colormap image if colorize=True."""
        if self.frames is None:
            return None
        depth_frame = self.frames.get_depth_frame()
        if depth_frame is None:
            return None

        w, h = depth_frame.get_width(), depth_frame.get_height()
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)
        depth_mm = depth.astype(np.float32) * depth_frame.get_depth_scale()

        if colorize:
            return cv.applyColorMap(
                cv.convertScaleAbs(depth_mm, alpha=0.05), cv.COLORMAP_JET
            )
        return depth_mm

    def release(self):
        self.pipeline.stop()
        cv.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def main():
    print("Orbbec live stream. Press 'q' to exit.")
    with Camera(cam_id=0) as cam:
        while True:
            if not cam.grab():
                continue

            color = cam.read_color()
            depth_vis = cam.read_depth(colorize=True)

            if color is not None:
                cv.imshow("Color", color)
            # if depth_vis is not None:
            # cv.imshow("Depth", depth_vis)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        cam.release()


if __name__ == "__main__":
    main()
