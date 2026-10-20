"""
ChArUco board: a fixed reference on the table, so part positions can be reported in a
board frame that does not change when the camera is moved.

Board frame (OpenCV convention): origin at the board's top-left outer corner as printed,
X along the long edge (squares_x), Y along the short edge, Z into the table. Height above
the board is therefore -Z.

    config/charuco_board.json   board definition (tracked); edit square_mm after measuring the print
    config/board_pose.json      camera -> board pose of the current setup (local, from `calibrate-board`)
"""

import json
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from vision_intelligence import paths

BOARD_SPEC = paths.CONFIG / "charuco_board.json"
BOARD_POSE = paths.CONFIG / "board_pose.json"
PRINT_DPI = 300
A3_MM = (420.0, 297.0)
MIN_CORNERS = 6  # fewest ChArUco corners accepted for a pose


def load_spec(path=BOARD_SPEC):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_board(spec):
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, spec["dictionary"]))
    return cv.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]), spec["square_mm"],
                                 spec["marker_mm"], dictionary)


def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], float)


# ---------------------------------------------------------------- printing

def render_print_files(spec=None, out_dir=paths.CONFIG, dpi=PRINT_DPI):
    """Write a print-ready A3 page (PNG and PDF, true scale at ``dpi``) with the board and a 100 mm scale bar."""
    from PIL import Image  # installed with ultralytics

    spec = spec or load_spec()
    board = make_board(spec)
    px_per_mm = dpi / 25.4
    board_mm = (spec["squares_x"] * spec["square_mm"], spec["squares_y"] * spec["square_mm"])
    board_px = (round(board_mm[0] * px_per_mm), round(board_mm[1] * px_per_mm))
    board_img = board.generateImage(board_px, marginSize=0, borderBits=1)

    page_px = (round(A3_MM[0] * px_per_mm), round(A3_MM[1] * px_per_mm))
    page = np.full((page_px[1], page_px[0]), 255, np.uint8)
    x0, y0 = (page_px[0] - board_px[0]) // 2, round(15 * px_per_mm)
    page[y0:y0 + board_px[1], x0:x0 + board_px[0]] = board_img

    # Scale bar (100 mm) and label below the board, to check the print scale
    bar_y = y0 + board_px[1] + round(10 * px_per_mm)
    bar_len, bar_h = round(100 * px_per_mm), round(3 * px_per_mm)
    page[bar_y:bar_y + bar_h, x0:x0 + bar_len] = 0
    text = (f"ChArUco {spec['dictionary']}  {spec['squares_x']}x{spec['squares_y']} squares  "
            f"square {spec['square_mm']:g} mm  marker {spec['marker_mm']:g} mm  |  black bar = 100 mm  |  "
            f"print at 100% (actual size), keep flat")
    cv.putText(page, text, (x0, bar_y + bar_h + round(9 * px_per_mm)), cv.FONT_HERSHEY_SIMPLEX,
               1.6, 0, 3, cv.LINE_AA)
    cv.putText(page, "origin (0,0)", (x0, y0 - round(3 * px_per_mm)), cv.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv.LINE_AA)

    out_dir = Path(out_dir)
    name = f"charuco_{spec['squares_x']}x{spec['squares_y']}_{spec['square_mm']:g}mm_A3"
    png, pdf = out_dir / f"{name}.png", out_dir / f"{name}.pdf"
    image = Image.fromarray(page)
    image.save(png, dpi=(dpi, dpi))
    image.save(pdf, resolution=dpi)
    return png, pdf


# ---------------------------------------------------------------- detection and pose

def detect_pose(image, intrinsics, spec=None, dist=None):
    """
    Board pose in one image. Returns dict(rvec, tvec (mm), corners, reprojection_px) or None
    if fewer than MIN_CORNERS ChArUco corners are found.
    """
    spec = spec or load_spec()
    board = make_board(spec)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _, _ = cv.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    K, dist = camera_matrix(intrinsics), np.zeros(5) if dist is None else np.asarray(dist, float)
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None
    projected, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2, axis=1))))
    return {"rvec": rvec.ravel(), "tvec": tvec.ravel(), "corners": int(len(ids)), "reprojection_px": error}


def save_pose(pose, intrinsics, path=BOARD_POSE, frames=1):
    data = {
        "rvec": [float(v) for v in pose["rvec"]], "tvec_mm": [float(v) for v in pose["tvec"]],
        "corners": pose["corners"], "reprojection_px": round(pose["reprojection_px"], 3), "frames_used": frames,
        "intrinsics": {k: getattr(intrinsics, k) for k in ("fx", "fy", "cx", "cy", "width", "height")},
        "board": load_spec(), "created": datetime.now().isoformat(timespec="seconds"),
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return Path(path)


def load_pose(path=BOARD_POSE):
    """(R, t) camera->board from config/board_pose.json, or None if the board was never calibrated."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    R, _ = cv.Rodrigues(np.array(data["rvec"], float))
    return R, np.array(data["tvec_mm"], float)


def calibrate_live(n_frames=30, max_reprojection_px=1.0, timeout_s=60):
    """
    Measure the camera -> board pose with the Orbbec camera: collect ``n_frames`` good board
    detections, take the median pose, save it to config/board_pose.json. The board must be
    fully visible and the camera still. Press 'q' to cancel.
    """
    import time

    from vision_intelligence.camera import Camera
    from vision_intelligence.visualization import put_text_lines

    spec = load_spec()
    detector = cv.aruco.CharucoDetector(make_board(spec))
    poses, camera = [], None
    try:
        camera = Camera()
        camera.start()
        start = time.monotonic()
        while len(poses) < n_frames and time.monotonic() - start < timeout_s:
            result = camera.get_frames()
            if result is None:
                continue
            color, _ = result
            pose = detect_pose(color, camera.intrinsics, spec)
            if pose and pose["reprojection_px"] <= max_reprojection_px:
                poses.append(pose)
            view = color.copy()
            corners, ids, _, _ = detector.detectBoard(cv.cvtColor(color, cv.COLOR_BGR2GRAY))
            if ids is not None:
                cv.aruco.drawDetectedCornersCharuco(view, corners, ids, (0, 255, 0))
            status = (f"corners {pose['corners']}, reprojection {pose['reprojection_px']:.2f}px" if pose
                      else "board not found")
            put_text_lines(view, [f"board calibration: {len(poses)}/{n_frames} good frames", status, "q = cancel"])
            cv.imshow("Board calibration", view)
            if cv.waitKey(1) & 0xFF == ord("q"):
                print("Cancelled")
                return None
        intrinsics = camera.intrinsics
    finally:
        if camera is not None:
            camera.release()
        cv.destroyAllWindows()

    if len(poses) < max(3, n_frames // 3):
        print(f"Only {len(poses)} good detections in {timeout_s}s: make the whole board visible, avoid glare, "
              "keep the camera still, then try again")
        return None
    tvecs = np.array([p["tvec"] for p in poses])
    best = dict(min(poses, key=lambda p: p["reprojection_px"]))
    best["tvec"] = np.median(tvecs, axis=0)
    best["rvec"] = np.median(np.array([p["rvec"] for p in poses]), axis=0)
    path = save_pose(best, intrinsics, frames=len(poses))
    spread = tvecs.std(axis=0)
    print(f"Saved {paths.to_root_relative(path)}: {len(poses)} frames, board origin at "
          f"{np.round(best['tvec']).astype(int)} mm from the camera, frame-to-frame spread "
          f"{np.round(spread, 1)} mm (x, y, z)")
    return path


def camera_to_board(points_mm, pose):
    """
    Camera-frame points (N x 3, mm) to board-frame points. ``pose`` = (R, t) with
    p_camera = R @ p_board + t, so p_board = R.T @ (p_camera - t). Height above the board = -Z.
    """
    R, t = pose
    pts = np.atleast_2d(np.asarray(points_mm, float))
    return (pts - t) @ R  # row-vector form of R.T @ (p - t)
