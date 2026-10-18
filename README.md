# Vision Intelligence

Object detection, localisation and orientation estimation using an Orbbec Gemini 2 RGB-D camera.

**Current status:** the camera pipeline is working. The app streams colour and depth (depth aligned to the colour image, in mm) and shows the 3D position of any pixel under the mouse. A YOLO-OBB detector for filters and frames can be trained on a labelled RGB dataset; 3D localisation and orientation from the detections are not implemented yet.

## Requirements

- Windows 10/11
- Python 3.13
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- Orbbec Gemini 2 camera connected to a **USB 3.0** port
- For training: an NVIDIA GPU is strongly recommended (tested on an RTX 4070 Laptop GPU, 8 GB)

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

### Where files are read and written

All commands find the project folder automatically (the folder with this `pyproject.toml`), so they work from any directory and on any machine; default data, model and output paths are always inside the project. Paths stored in `data/datasets/index.csv` are relative to the project folder. To use a different location, set the environment variable `VISION_INTELLIGENCE_ROOT`, e.g. in PowerShell `$env:VISION_INTELLIGENCE_ROOT = "D:\vision-data"`. Paths you pass on the command line (e.g. `--out`) are relative to your current folder.

On Windows and Linux, `uv sync` installs PyTorch with CUDA 12.8 (NVIDIA GPU); on macOS it installs the standard build, and training uses the Apple GPU (`mps`) or CPU.

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

## Training the detector

Trains Ultralytics YOLO-OBB (`yolo11n-obb`) on a labelled dataset. Training uses the NVIDIA GPU automatically if one is available (PyTorch is installed with CUDA 12.8 by `uv sync`).

Labels are YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized). Class names come from the dataset YAML (current scheme: `0 = Air_Filter`, `1 = filter`, `2 = frame`); evaluate, test and detect use the trained model's own class names, so any scheme works.

### 1. Split the labelled data (recommended)

Consecutive camera frames are almost identical, so a random split puts near-duplicates of test images into train and inflates the test scores. `split` pools all images of a dataset (e.g. a Roboflow export) and re-splits each capture session into contiguous time blocks (70 / 15 / 15 %), dropping a few frames at each boundary:

```
uv run python -m vision_intelligence split --data data/data.yaml            # -> data/datasets/rgb/
uv run python -m vision_intelligence split --data data/data.yaml --gap 5 --overwrite
```

Image names must start with `<session>_<frame>` (e.g. `20261016_111506_0042...`). Augmented copies of a frame stay together; val/test keep a single copy. The output has `images/{train,val,test}/`, `labels/...`, `data.yaml` and `split_report.csv`. The input is only copied, never changed.

### 2. Train

```
uv run python -m vision_intelligence train --data data/datasets/rgb/data.yaml   # after `split`
uv run python -m vision_intelligence train --data data/data.yaml                # Roboflow export as it is
uv run python -m vision_intelligence train                                      # data/datasets/rgb/ layout
uv run python -m vision_intelligence train --data ... --seeds 0 1 2             # three runs with different seeds
```

`--data` accepts any Ultralytics dataset YAML, including Roboflow's `{train,valid,test}/{images,labels}` layout. Before training the script prints the classes and image counts per split, and warns if `imgsz` does not match the image size.

Settings (same for RGB and RGB-D, defined in `COMMON` in `src/vision_intelligence/detection/train.py`): 200 epochs, image size 640 (set this to your images' size), batch 8, early stop after 30 epochs without improvement, colour (HSV) augmentation off. `--epochs`, `--imgsz` and `--batch` exist for quick tests only; never change them between the RGB and RGB-D model. The dataset and image size of each run are saved in `model/<run>/args.yaml`, and `evaluate`, `test` and `detect` reuse them automatically.

Output:

```
data/datasets/rgb.yaml                 # dataset config written by the script
model/pretrained/yolo11n-obb.pt        # pretrained start, downloaded on first run
model/rgb_s0/weights/best.pt           # trained model (best validation mAP); use this for prediction
model/rgb_s0/weights/last.pt           # model after the last epoch
model/rgb_s0/results.csv, *.png        # training curves, confusion matrix, example predictions
```

## Evaluating and testing a trained model

Do this before trying the model live.

**Scores** on the held-out `test` split (and `test_white` if it exists) for every trained run in `model/`:

```
uv run python -m vision_intelligence evaluate                 # all runs, each on its own training dataset's test split
uv run python -m vision_intelligence evaluate --runs rgb_s0   # one run
uv run python -m vision_intelligence evaluate --data other/data.yaml   # test on a different dataset
```

Prints a table (mAP50, mAP50-95, precision / recall / AP50 for every class, inference time per image), writes `model/comparison.csv`, and saves Ultralytics plots (confusion matrix, PR curves) to `model/<run>/eval_test/`. With several seeds it also prints the mean and standard deviation. Filter recall on `test_white` is the primary metric of the RGB vs RGB-D comparison.

**Visual test**: predictions drawn on the test images next to the ground-truth labels:

```
uv run python -m vision_intelligence test                                       # model/rgb_s0 on its test split
uv run python -m vision_intelligence test --source path/to/new/images           # any image or folder
uv run python -m vision_intelligence test --weights model/rgb_s1/weights/best.pt --conf 0.3
```

Output in `model/<run>/test_<folder>/`: one image per input (orange = filter, magenta = frame, cyan = Air_Filter, green = ground truth, red = missed), `overview.png` with all images as numbered tiles, `results.csv` (per image: correct, missed and false detections, `OK` or `CHECK`) and `predictions.json` (class, confidence, corners, approximate 3D centre, yaw). A prediction counts as correct if it has the right class and overlaps the ground truth by IoU ≥ 0.5 (`--iou`).

## Live detection test

Runs the trained model on the camera stream:

```
uv run python -m vision_intelligence detect                     # uses model/rgb_s0/weights/best.pt
uv run python -m vision_intelligence detect --conf 0.3          # also show less confident detections
uv run python -m vision_intelligence detect --weights model/rgbd_s0/weights/best.pt
uv run python -m vision_intelligence detect --stretch           # only if the training images were stretched to a square
```

The camera gives 1280×720 frames. By default they are letterboxed (aspect ratio kept) to the model's training size. If the training images were made with Roboflow's "Resize: Stretch to 640x640", use `--stretch` so live frames are squashed the same way; otherwise parts look a different shape than in training.

| Key | Effect |
|---|---|
| `s` | Save a snapshot (raw colour + annotated image) to `outputs/live/` |
| `q` | Quit |

Each detection shows class, confidence, yaw (degrees, -90 to 90, 0 = along the image x axis, positive = clockwise) and an approximate 3D centre in mm in the camera frame. The depth is taken at the box centre, so for a hollow frame it is the surface below the frame.

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
