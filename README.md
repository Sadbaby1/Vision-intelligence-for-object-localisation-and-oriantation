# Vision Intelligence

Object detection, localisation and orientation estimation using an Orbbec Gemini 2 RGB-D camera.

**Current status:** the camera pipeline is working. The app streams colour and depth (depth aligned to the colour image, in mm) and shows the 3D position of any pixel under the mouse. Object detection and orientation estimation are not implemented yet.

## Requirements

- Windows 10/11
- Python 3.13
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- Orbbec Gemini 2 camera connected to a **USB 3.0** port

## Installation

1. **Install uv** (one time):

   ```
   winget install astral-sh.uv
   ```

   Then fully close and reopen VS Code / your terminal so `uv` is on the PATH. Check with `uv --version`.

2. **Clone the repository:**

   ```
   git clone https://github.com/Sadbaby1/Vision-intelligence-for-object-localisation-and-oriantation.git
   cd Vision-intelligence-for-object-localisation-and-oriantation
   ```

3. **Create the virtual environment and install dependencies:**

   ```
   uv sync
   ```

   This creates `.venv/` and installs everything listed in `pyproject.toml` (OpenCV, Orbbec SDK, and the project itself). Do not copy `.venv/` between computers; run `uv sync` on each machine instead.

## Running

Connect the camera, then from the project folder:

```
uv run vision-intelligence
```

If the virtual environment is already activated in your terminal, `vision-intelligence` on its own also works.

### Controls

| Key / action | Effect |
|---|---|
| `v` | Switch view: overlay → color → depth |
| Mouse hover | Show the 3D position (X, Y, Z) in mm of the point under the cursor |
| `q` | Quit |

The 3D position is in the camera frame: X to the right, Y down, Z forward from the camera, in millimetres.

### Checking that the camera works

1. **Depth is received:** the top-left text shows `valid depth: …%`. Around 70% or more is normal.
2. **Depth is aligned with colour:** in the overlay view, the depth colours should sit exactly on the objects in the colour image.
3. **Distance is correct:** hover over an object and compare Z with a tape measure from the front of the camera. They should agree within about 1–2%.

Glass, shiny or black surfaces, and anything closer than about 0.2 m, often have no depth. This is a limitation of the sensor, not a bug.

## Capturing training images

Use `vision-capture` to take colour + depth pictures for building a YOLO dataset:

```
uv run vision-capture                  # manual: press SPACE for each picture
uv run vision-capture --interval 2     # also allows auto mode: press 'a' to take a picture every 2 s
uv run vision-capture --out D:\datasets\captures   # save somewhere else
```

| Key | Effect |
|---|---|
| `SPACE` | Take a picture |
| `a` | Toggle automatic capture (needs `--interval`) |
| `q` | Quit |

The preview shows colour and depth side by side. The screen flashes briefly each time a picture is saved.

Each run creates a new session folder (default `data/captures/`, which is not committed to git):

```
data/captures/20261010_142501/
├── color/20261010_142501_0001.png   # colour image: use these for labelling / YOLO training
├── depth/20261010_142501_0001.png   # 16-bit depth in mm, aligned to the colour image (0 = no depth)
└── intrinsics.json                  # camera intrinsics for this session
```

Colour and depth files with the same name were taken at the same moment. To read a depth image in Python:

```python
depth_mm = cv2.imread("depth/20261010_142501_0001.png", cv2.IMREAD_UNCHANGED)  # uint16, mm
```

Tips for a good dataset: vary object position, rotation, distance (about 0.4–1.5 m), background and lighting, and include several objects in the same picture.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv : The term 'uv' is not recognized` | The terminal was opened before uv was installed. In VS Code run **Terminal: Kill All Terminals**, then **File → Exit**, and reopen. |
| `Hardware MFT failed ... lack of hardware resources` | Another program (a second copy of the app, Orbbec Viewer) is using the camera. Close it and try again. |
| `No Orbbec camera found` | Check the USB cable and that the camera is on a USB 3.0 port. |
| `VIRTUAL_ENV=...anaconda3 does not match...` warning | Harmless; uv ignores the conda environment. Run `conda deactivate` to hide it. |

## Project structure

```
Vision-Intelligence/
├── src/vision_intelligence/
│   ├── app.py                  # Main loop: live view, depth overlay, 3D readout
│   ├── capture.py              # `vision-capture`: save colour + depth images for datasets
│   ├── visualization.py        # Depth colouring and on-screen text helpers
│   ├── __main__.py             # Enables `python -m vision_intelligence`
│   ├── camera/
│   │   ├── orbbec.py           # Camera: color + depth streams, depth aligned to color, intrinsics
│   │   └── frame_utils.py      # Orbbec frame format -> BGR image conversion
│   ├── detection/
│   │   ├── detector.py         # Object detection model (to do)
│   │   └── detected_object.py  # Detection result data class (to do)
│   └── geometry/
│       └── coordinate.py       # Pixel + depth -> 3D point in the camera frame
├── experiments/                # Scratch scripts, not part of the package
├── tests/
└── pyproject.toml              # Dependencies and the `vision-intelligence` / `vision-capture` commands
```

## Adding dependencies

```
uv add <package-name>
```

This updates `pyproject.toml` and `uv.lock`. Commit both files.
