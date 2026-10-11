import cv2 as cv
import numpy as np

from vision_intelligence.camera import Camera
from vision_intelligence.geometry.coordinate import deproject_pixel, depth_at

# import for detection and orientation
from vision_intelligence.detection.detector import Detector
from vision_intelligence.detection.aruco import ArucoCalibrator
from vision_intelligence.detection.detected_object import DetectedObject
from vision_intelligence.orientation.orientation import OrientationEstimator
from vision_intelligence.geometry.coordinate import CoordinateTransformer

WINDOW = "Vision Intelligence"
MAX_DISPLAY_WIDTH = 1280
DEPTH_VIS_RANGE_MM = (200, 3000)
VIEWS = ("overlay", "color", "depth")


def colorize_depth(depth_mm):
    """Map depth to a JET colour image; pixels with no depth are drawn black."""
    near, far = DEPTH_VIS_RANGE_MM
    scaled = np.clip((depth_mm - near) / (far - near), 0, 1)
    vis = cv.applyColorMap((scaled * 255).astype(np.uint8), cv.COLORMAP_JET)
    vis[depth_mm <= 0] = 0
    return vis


def process_frame(frame, calibrator, detector):
    """Process a single frame from the Orbbec camera."""

    # Find ArUco markers and calculate pixel-to-mm ratio
    bbox, ids = calibrator.find_markers(frame)
    ratio = calibrator.calculate_ratio(bbox)
    # Get marker's pixel center and build the transformer for this frame
    marker_center = calibrator.get_marker_center(bbox)

    # Run YOLOv8 detection on the frame
    detected_objects = detector.detect(frame)

    if ratio is None or marker_center is None:
        print("No marker detected.")
        return

    print(f"Pixel-to-mm ratio: {ratio:.4f} pixels/mm")

    transformer = CoordinateTransformer(
        marker_world_x=148.5, marker_world_y=210, pixel_mm_ratio=ratio
    )

    # center of detected ArUco marker in pixels
    marker_center = calibrator.get_marker_center(bbox)

    # Convert each detected object's pixel position into robot coordinates
    for obj in detected_objects:
        object_pixel_x, object_pixel_y = obj.center()
        robot_x, robot_y = transformer.transform(
            object_pixel_x, object_pixel_y, marker_center[0], marker_center[1]
        )
        print(
            f"{obj.label} → robot coords: ({robot_x:.1f}, {robot_y:.1f}) mm, orientation={obj.orientation}"
        )


def main():

    orbbec = None
    print("Starting Orbbec camera...")

    # Mouse position in full-resolution image coordinates
    mouse = {"u": None, "v": None, "scale": 1.0}

    def on_mouse(event, x, y, flags, param):
        mouse["u"] = int(x / mouse["scale"])
        mouse["v"] = int(y / mouse["scale"])

    try:
        orbbec = Camera()
        orbbec.start()
        calibrator = ArucoCalibrator(marker_length_mm=100)
        detector = Detector(model_path="best.pt")

        cv.namedWindow(WINDOW)
        cv.setMouseCallback(WINDOW, on_mouse)
        view = 0
        intrinsics_printed = False

        print("Live stream. 'v' = switch view, hover = depth/XYZ, 'q' = quit.")
        while True:
            result = orbbec.get_frames()
            if result is None:
                continue
            color_frame, depth_frame = result
            # sadate code
            process_frame(color_frame, calibrator, detector)

            if not intrinsics_printed:
                k = orbbec.intrinsics
                print(
                    f"Color {color_frame.shape[1]}x{color_frame.shape[0]}, "
                    f"intrinsics fx={k.fx:.1f} fy={k.fy:.1f} cx={k.cx:.1f} cy={k.cy:.1f}"
                )
                intrinsics_printed = True

            depth_vis = colorize_depth(depth_frame)
            if VIEWS[view] == "color":
                frame = color_frame.copy()

            elif VIEWS[view] == "depth":
                frame = depth_vis
            else:
                frame = cv.addWeighted(color_frame, 0.5, depth_vis, 0.5, 0)

            valid_pct = (depth_frame > 0).mean() * 100
            lines = [f"view: {VIEWS[view]}   valid depth: {valid_pct:.0f}%"]

            u, v = mouse["u"], mouse["v"]
            h, w = depth_frame.shape
            if u is not None and 0 <= u < w and 0 <= v < h:
                cv.drawMarker(frame, (u, v), (255, 255, 255), cv.MARKER_CROSS, 30, 2)
                z = depth_at(depth_frame, u, v)
                if z is None:
                    lines.append(f"pixel ({u},{v}): no depth")
                else:
                    x, y, z = deproject_pixel(u, v, z, orbbec.intrinsics)
                    lines.append(
                        f"pixel ({u},{v}): XYZ = ({x:.0f}, {y:.0f}, {z:.0f}) mm"
                    )

            scale = min(1.0, MAX_DISPLAY_WIDTH / w)
            mouse["scale"] = scale
            display = (
                cv.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
            )
            for i, text in enumerate(lines):
                org = (10, 25 + i * 25)
                cv.putText(
                    display, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4
                )
                cv.putText(
                    display, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )

            cv.imshow(WINDOW, display)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                view = (view + 1) % len(VIEWS)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if orbbec is not None:
            orbbec.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
