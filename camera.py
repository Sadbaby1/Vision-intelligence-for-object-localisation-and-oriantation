import numpy as np
import cv2 as cv
from pyorbbecsdk import (
    Device,
    Context,
    Pipeline,
    Config,
    AlignFilter,
    OBSensorType,
    OBStreamType,
    OBFrameAggregateOutputMode,
)

from utils import frame_to_bgr_image


class Camera:
    def __init__(self):

        self.started = False
        self.intrinsics = None

        # Check for a device BEFORE creating the Pipeline
        ctx = Context()
        devices = ctx.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("No Orbbec camera found")

        self.pipeline = Pipeline()
        self.config = Config()

        # Align depth pixels with RGB pixels
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def start(self):

        if self.started:
            raise RuntimeError("Camera already started")

        color_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        color_profile = color_profiles.get_default_video_stream_profile()

        depth_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        depth_profile = depth_profiles.get_default_video_stream_profile()

        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)

        self.config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )

        self.pipeline.start(self.config)
        self.started = True

        print("Gemini 2 started")

    def get_frames(self):
        if not self.started:
            raise RuntimeError("Camera is not started")

        frames = self.pipeline.wait_for_frames(1000)

        if frames is None:
            return None

        # Software depth-to-color alignment
        aligned = self.align_filter.process(frames)

        if aligned is None:
            return None

        if hasattr(aligned, "as_frame_set"):
            aligned = aligned.as_frame_set()

        if aligned is None:
            return None

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None

        color = frame_to_bgr_image(color_frame)

        if color is None:
            return None

        h = depth_frame.get_height()
        w = depth_frame.get_width()

        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(h, w)

        # Convert raw values into millimeters
        depth_mm = depth_raw.astype(np.float32) * depth_frame.get_depth_scale()

        # Alignment must produce matching dimensions
        if depth_mm.shape != color.shape[:2]:
            raise RuntimeError("Aligned depth and RGB sizes do not match")

        # Get intrinsics for the RGB coordinate system
        profile = color_frame.get_stream_profile()
        self.intrinsics = profile.get_intrinsic()

        return color, depth_mm

    def release(self):
        if self.started:
            try:
                self.pipeline.stop()

            finally:
                self.started = False
        print("orbbec stopped")
