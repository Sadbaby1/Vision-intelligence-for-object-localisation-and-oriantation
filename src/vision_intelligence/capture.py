"""
Capture aligned colour + depth image pairs for building a training dataset.

Each session is written to its own folder:

    <out>/<session>/
        color/<session>_0001.png   # BGR image, use this for YOLO labelling
        depth/<session>_0001.png   # uint16 PNG, depth in mm aligned to the colour image (0 = no depth)
        intrinsics.json            # colour camera intrinsics for this session

Colour and depth files with the same name belong to the same moment.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from vision_intelligence import paths
from vision_intelligence.camera import Camera
from vision_intelligence.visualization import colorize_depth, put_text_lines

WINDOW = "Vision Capture"
DEFAULT_OUT_DIR = paths.CAPTURES
PREVIEW_SIZE = (640, 360)


class CaptureSession:
    """Writes colour/depth pairs into a new timestamped session folder."""

    def __init__(self, out_dir=DEFAULT_OUT_DIR):
        self.name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(out_dir) / self.name
        self.color_dir = self.dir / "color"
        self.depth_dir = self.dir / "depth"
        self.count = 0
        self.intrinsics_saved = False

    def save(self, color, depth_mm, intrinsics=None):
        """Save one colour/depth pair. Returns the path of the colour image."""
        self.color_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        if intrinsics is not None and not self.intrinsics_saved:
            self._save_intrinsics(intrinsics)

        self.count += 1
        filename = f"{self.name}_{self.count:04d}.png"

        # Round to whole millimetres; uint16 covers 0-65 m, far beyond the sensor range
        depth_u16 = np.clip(np.round(depth_mm), 0, 65535).astype(np.uint16)

        color_path = self.color_dir / filename
        if not cv.imwrite(str(color_path), color):
            raise RuntimeError(f"Failed to write {color_path}")
        if not cv.imwrite(str(self.depth_dir / filename), depth_u16):
            raise RuntimeError(f"Failed to write {self.depth_dir / filename}")
        return color_path

    def _save_intrinsics(self, k):
        data = {
            "width": k.width,
            "height": k.height,
            "fx": k.fx,
            "fy": k.fy,
            "cx": k.cx,
            "cy": k.cy,
            "depth_unit": "mm",
        }
        (self.dir / "intrinsics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.intrinsics_saved = True


def main(default_out=DEFAULT_OUT_DIR):
    parser = argparse.ArgumentParser(description="Capture colour + depth images for a training dataset.")
    parser.add_argument("--out", type=Path, default=default_out, help=f"output folder (default: {default_out})")
    parser.add_argument(
        "--interval",
        type=float,
        default=0,
        help="seconds between automatic captures when auto mode is on (key 'a'); 0 disables auto mode",
    )
    args = parser.parse_args()

    orbbec = None
    session = CaptureSession(args.out)
    auto = False
    last_auto = 0.0
    flash_until = 0.0

    try:
        orbbec = Camera()
        orbbec.start()

        print(f"Saving to {session.dir.resolve()}")
        print("SPACE = capture, 'a' = toggle auto capture, 'q' = quit.")
        while True:
            result = orbbec.get_frames()
            if result is None:
                continue
            color, depth_mm = result

            now = time.monotonic()
            take = False
            if auto and now - last_auto >= args.interval:
                take = True
                last_auto = now

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                take = True
            if key == ord("a"):
                if args.interval > 0:
                    auto = not auto
                    last_auto = now
                else:
                    print("Auto capture needs --interval, e.g. vision-capture --interval 2")

            if take:
                path = session.save(color, depth_mm, orbbec.intrinsics)
                print(f"[{session.count}] saved {path.name}")
                flash_until = now + 0.15

            # Preview: colour | depth, side by side at reduced size
            preview = cv.hconcat(
                [cv.resize(color, PREVIEW_SIZE), cv.resize(colorize_depth(depth_mm), PREVIEW_SIZE)]
            )
            valid_pct = (depth_mm > 0).mean() * 100
            mode = f"AUTO every {args.interval:g}s" if auto else "manual"
            put_text_lines(
                preview,
                [f"saved: {session.count}   mode: {mode}", f"valid depth: {valid_pct:.0f}%"],
            )
            if now < flash_until:
                preview = cv.bitwise_not(preview)
            cv.imshow(WINDOW, preview)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if orbbec is not None:
            orbbec.release()
        cv.destroyAllWindows()
        print(f"Captured {session.count} image pairs in {session.dir.resolve()}")


def main_frame():
    main(DEFAULT_OUT_DIR / "frames")


def main_filter():
    main(DEFAULT_OUT_DIR / "filters")

def main_assembled():
    main(DEFAULT_OUT_DIR / "assembled")

def main_completed():
    main(DEFAULT_OUT_DIR / "completed")

if __name__ == "__main__":
    main()
