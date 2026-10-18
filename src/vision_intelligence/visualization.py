import cv2 as cv
import numpy as np

DEPTH_VIS_RANGE_MM = (200, 3000)


def colorize_depth(depth_mm, vis_range_mm=DEPTH_VIS_RANGE_MM):
    """Map depth to a JET colour image; pixels with no depth are drawn black."""
    near, far = vis_range_mm
    scaled = np.clip((depth_mm - near) / (far - near), 0, 1)
    vis = cv.applyColorMap((scaled * 255).astype(np.uint8), cv.COLORMAP_JET)
    vis[depth_mm <= 0] = 0
    return vis


def tile_overview(images, cols=5, tile=(384, 216)):
    """Grid of small numbered tiles (1, 2, ...) for a quick visual check of many images."""
    tiles = []
    for number, img in enumerate(images, 1):
        t = cv.resize(img, tile)
        cv.putText(t, str(number), (8, 40), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6)
        cv.putText(t, str(number), (8, 40), cv.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return cv.vconcat([cv.hconcat(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def put_text_lines(img, lines, origin=(10, 25), line_height=25):
    """Draw white text with a dark outline so it is readable on any background."""
    x, y = origin
    for i, text in enumerate(lines):
        org = (x, y + i * line_height)
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
