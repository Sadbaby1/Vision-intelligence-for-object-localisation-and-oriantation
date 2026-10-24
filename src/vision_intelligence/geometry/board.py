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


# ---------------------------------------------------------------- board plane from the depth sensor
#
# The ChArUco pose comes from the colour image; part depths come from the depth sensor. A constant
# depth-sensor offset (typically 1-3 % at 1 m) would then appear as an error in "height above the
# board". Measuring the board plane with the depth sensor as well makes the height a difference
# of two depth measurements, so such an offset cancels.

def board_region(shape, intrinsics, R, t, spec=None, inset_mm=15.0):
    """Mask of the board area in the image (the printed board, shrunk by ``inset_mm``)."""
    spec = spec or load_spec()
    w, h = spec["squares_x"] * spec["square_mm"], spec["squares_y"] * spec["square_mm"]
    corners = np.array([[inset_mm, inset_mm, 0], [w - inset_mm, inset_mm, 0],
                        [w - inset_mm, h - inset_mm, 0], [inset_mm, h - inset_mm, 0]], float)
    pts, _ = cv.projectPoints(corners, cv.Rodrigues(R)[0], t, camera_matrix(intrinsics), np.zeros(5))
    mask = np.zeros(shape, np.uint8)
    cv.fillPoly(mask, [np.round(pts.reshape(-1, 2)).astype(np.int32)], 1)
    return mask


def _points_from_depth(depth_mm, intrinsics, mask, step=3):
    v, u = np.nonzero(mask[::step, ::step])
    v, u = v * step, u * step
    z = depth_mm[v, u].astype(float)
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    return np.c_[(u - intrinsics.cx) * z / intrinsics.fx, (v - intrinsics.cy) * z / intrinsics.fy, z]


def fit_plane(points, inlier_mm=5.0):
    """Least-squares plane n.p = d (unit n pointing away from the camera) with one outlier-rejection pass."""
    def fit(p):
        c = p.mean(axis=0)
        n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
        n = -n if n[2] < 0 else n
        return n, float(n @ c)

    n, d = fit(points)
    inliers = points[np.abs(points @ n - d) < inlier_mm]
    if len(inliers) >= 50:
        n, d = fit(inliers)
    rms = float(np.sqrt(np.mean((inliers @ n - d) ** 2))) if len(inliers) else float("nan")
    return n, d, rms, int(len(inliers))


def pnp_plane(R, t):
    """Board plane from the ChArUco pose, as n.p = d with n pointing away from the camera."""
    n = R[:, 2] if R[2, 2] > 0 else -R[:, 2]
    return n, float(n @ t)


def depth_minus_pnp_mm(depth_mm, intrinsics, mask, R, t):
    """Median of (measured depth - depth predicted by the ChArUco pose) over the board area, in mm."""
    n, d = pnp_plane(R, t)
    v, u = np.nonzero(mask)
    z = depth_mm[v, u].astype(float)
    ok = z > 0
    rays = np.c_[(u[ok] - intrinsics.cx) / intrinsics.fx, (v[ok] - intrinsics.cy) / intrinsics.fy, np.ones(ok.sum())]
    z_pred = d / (rays @ n)  # ray-plane intersection, z component of the ray is 1
    return float(np.median(z[ok] - z_pred))


def height_above_plane(points_mm, plane):
    """Height (mm) of camera-frame points above a plane n.p = d (n away from the camera)."""
    n, d = plane
    return d - np.atleast_2d(np.asarray(points_mm, float)) @ np.asarray(n, float)


def save_pose(pose, intrinsics, path=BOARD_POSE, frames=1, depth_plane=None):
    data = {
        "rvec": [float(v) for v in pose["rvec"]], "tvec_mm": [float(v) for v in pose["tvec"]],
        "corners": pose["corners"], "reprojection_px": round(pose["reprojection_px"], 3), "frames_used": frames,
        "intrinsics": {k: getattr(intrinsics, k) for k in ("fx", "fy", "cx", "cy", "width", "height")},
        "board": load_spec(), "created": datetime.now().isoformat(timespec="seconds"),
    }
    if depth_plane:
        data["depth_plane"] = depth_plane
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return Path(path)


def load_depth_plane(path=BOARD_POSE):
    """(n, d) of the board plane measured by the depth sensor, or None (older calibration / no board)."""
    path = Path(path)
    if not path.exists():
        return None
    plane = json.loads(path.read_text(encoding="utf-8")).get("depth_plane")
    return (np.array(plane["normal"], float), float(plane["d_mm"])) if plane else None


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
    detections, take the median pose, and also fit the board plane in the depth image of the
    same frames; save both to config/board_pose.json. The board must be fully visible and the
    camera still. Press 'q' to cancel.
    """
    import time

    from vision_intelligence.camera import Camera
    from vision_intelligence.visualization import put_text_lines

    spec = load_spec()
    detector = cv.aruco.CharucoDetector(make_board(spec))
    poses, planes, offsets, camera = [], [], [], None
    try:
        camera = Camera()
        camera.start()
        start = time.monotonic()
        while len(poses) < n_frames and time.monotonic() - start < timeout_s:
            result = camera.get_frames()
            if result is None:
                continue
            color, depth_mm = result
            pose = detect_pose(color, camera.intrinsics, spec)
            if pose and pose["reprojection_px"] <= max_reprojection_px:
                poses.append(pose)
                # the same board, measured by the depth sensor
                R = cv.Rodrigues(pose["rvec"])[0]
                mask = board_region(depth_mm.shape, camera.intrinsics, R, pose["tvec"], spec)
                pts = _points_from_depth(depth_mm, camera.intrinsics, mask)
                if len(pts) >= 200:
                    planes.append(fit_plane(pts))
                    offsets.append(depth_minus_pnp_mm(depth_mm, camera.intrinsics, mask, R, pose["tvec"]))
            view = color.copy()
            corners, ids, _, _ = detector.detectBoard(cv.cvtColor(color, cv.COLOR_BGR2GRAY))
            if ids is not None and len(ids):
                # OpenCV 5 returns (N, 2) corners and (N,) ids; the drawing function needs (N, 1, 2) and (N, 1)
                cv.aruco.drawDetectedCornersCharuco(view, corners.reshape(-1, 1, 2), ids.reshape(-1, 1), (0, 255, 0))
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
    depth_plane = None
    if planes:
        n = np.median(np.array([p[0] for p in planes]), axis=0)
        depth_plane = {"normal": [float(v) for v in n / np.linalg.norm(n)],
                       "d_mm": float(np.median([p[1] for p in planes])),
                       "rms_mm": round(float(np.median([p[2] for p in planes])), 2),
                       "points_per_frame": int(np.median([p[3] for p in planes])),
                       "depth_minus_charuco_mm": round(float(np.median(offsets)), 1), "frames": len(planes)}
    path = save_pose(best, intrinsics, frames=len(poses), depth_plane=depth_plane)
    spread = tvecs.std(axis=0)
    print(f"Saved {paths.to_root_relative(path)}: {len(poses)} frames, board origin at "
          f"{np.round(best['tvec']).astype(int)} mm from the camera, frame-to-frame spread "
          f"{np.round(spread, 1)} mm (x, y, z)")
    if depth_plane:
        off = depth_plane["depth_minus_charuco_mm"]
        print(f"Depth sensor vs ChArUco: the depth sensor measures the board {abs(off):.1f} mm "
              f"{'farther' if off > 0 else 'closer'} than the ChArUco pose (board flatness in depth: "
              f"{depth_plane['rms_mm']} mm rms). Heights above the board now use the depth-measured "
              "board plane, so this offset cancels out.")
    else:
        print("WARNING: no valid depth on the board; heights fall back to the ChArUco plane")
    return path


def camera_to_board(points_mm, pose):
    """
    Camera-frame points (N x 3, mm) to board-frame points. ``pose`` = (R, t) with
    p_camera = R @ p_board + t, so p_board = R.T @ (p_camera - t). Height above the board = -Z.
    """
    R, t = pose
    pts = np.atleast_2d(np.asarray(points_mm, float))
    return (pts - t) @ R  # row-vector form of R.T @ (p - t)
