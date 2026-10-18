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

## Datasets and models: how they are organised

Every dataset version and every trained model has its own folder, so experiments never mix and nothing has to be deleted to train something new:

```
data/
├── captures/<folder>/<session>/          raw camera data: color/, depth/, intrinsics.json   (read-only)
├── roboflow/<export>/                    Roboflow exports as downloaded                       (read-only)
└── datasets/<dataset_id>/                one folder per dataset version
    ├── manifest.json                     where the data came from, split settings, classes, counts
    ├── index.csv                         every image -> session, frame, split, source file
    ├── rgb/{images,labels}/{train,val,test}/ + data.yaml
    └── rgbd/...                          4-channel version of the same images (RGB-D model)
model/
├── pretrained/yolo11n-obb.pt
├── registry.csv                          one row per training run (list it with `runs`)
└── <dataset_id>/
    ├── comparison.csv                    `evaluate` results for all models of this dataset
    └── <variant>_<model>_s<seed>/        e.g. rgb_yolo11n-obb_s0
        ├── weights/best.pt, last.pt
        ├── run_info.json                 dataset, classes, channels, image size, settings, scores
        ├── args.yaml, results.csv, *.png Ultralytics training record and plots
        ├── eval_test/                    from `evaluate`
        └── test_test/                    from `test`
```

- `split` refuses to write into an existing dataset ID and `train` refuses an existing run; use a new ID / `--name`, or `--overwrite` on purpose.
- Use a new dataset ID for every new export or labelling round.
- Current content:

| What | Where | Notes |
|---|---|---|
| Roboflow export (as downloaded) | `data/roboflow/rgb_2026-10-16_export_1280/` | includes Roboflow-augmented copies, no depth |
| Same labels paired with colour + depth | `data/roboflow/rgbd_2026-10-16_export_1280/` | `pair-depth`: unchanged images only (682 frames) |
| Dataset v1 (RGB only) | `data/datasets/2026-10-16_rgb_v1/` | from the export incl. augmented copies |
| Dataset v2 (RGB and RGB-D) | `data/datasets/2026-10-16_v2/` | from the paired folder; `rgb/` and `rgbd/` share images, labels and split |
| Model trained on v1 | `model/2026-10-16_rgb_v1/rgb_yolo11n-obb_s0/` | |

## Training the detector

Trains Ultralytics YOLO-OBB (`yolo11n-obb`) on a labelled dataset. Training uses the NVIDIA GPU automatically if one is available (PyTorch is installed with CUDA 12.8 by `uv sync`).

Labels are YOLO OBB (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized). Class names come from the dataset YAML (current scheme: `0 = Air_Filter`, `1 = filter`, `2 = frame`); evaluate, test and detect use the trained model's own class names, so any scheme works.

### 1. Pair the labels with depth (for RGB-D)

Roboflow keeps the original file name (`20261016_111506_0001_png.rf.<hash>.jpg`). `pair-depth` matches every labelled export image to its original capture frame, keeps only images Roboflow did not change (no augmentation), and copies the original colour image, its depth image and the session intrinsics next to the label:

```
uv run python -m vision_intelligence pair-depth --data data/roboflow/<export>/data.yaml --out data/roboflow/rgbd_<export>
```

Output: `rgb/images`, `rgb/labels`, `depth/`, `intrinsics/`, `index.csv`, `manifest.json` and `check/` (labels drawn on colour and on depth, to verify the alignment).

### 2. Create a dataset version (train / val / test split)

Consecutive camera frames are almost identical, so a random split puts near-duplicates of test images into train and inflates the test scores. `split` re-splits each capture session into contiguous time blocks (70 / 15 / 15 %), dropping a few frames at each boundary. Both models later use this one split:

```
uv run python -m vision_intelligence split --data data/roboflow/rgbd_2026-10-16_export_1280/data.yaml --dataset 2026-10-16_v2
```

Image names must start with `<session>_<frame>`. Output: `data/datasets/<dataset_id>/` with `rgb/`, `index.csv` (incl. the depth and intrinsics file of every image, if paired) and `manifest.json`. The input is only copied, never changed.

### 3. Build the RGB-D version

```
uv run python -m vision_intelligence build-rgbd --dataset 2026-10-16_v2
```

Creates `<dataset_id>/rgbd/` with the same images, labels and split as `rgb/`, stored as 4-channel TIFFs (B, G, R, local height). Local height is the height of each pixel above its local surroundings, computed from depth (`geometry/depth.py`, parameters in `rgbd/rgbd_config.json`: kernel 151 px, full scale 40 mm, depth smoothing 9 px). `rgbd/preview/` holds the height channel as images: check that parts appear as bright shapes before training. Thin frames give almost no depth signal; the RGB-D model sees them through the colour channels.

### 4. Train

`--variant` says which version of the dataset is used:

```
uv run python -m vision_intelligence train --dataset 2026-10-16_v2 --variant rgb     # RGB model:   colour images (rgb/)
uv run python -m vision_intelligence train --dataset 2026-10-16_v2 --variant rgbd    # RGB-D model: colour + local height (rgbd/)
uv run python -m vision_intelligence train --dataset 2026-10-16_v2 --variant rgb --seeds 0 1 2   # three seeds
uv run python -m vision_intelligence runs                                            # list all trained runs
```

Before training the script prints the classes and image counts per split, and warns if `imgsz` does not match the image size. Weights go to `model/<dataset_id>/<variant>_<model>_s<seed>/weights/best.pt`.

Settings (same for RGB and RGB-D, defined in `COMMON` in `src/vision_intelligence/detection/train.py`): 200 epochs, image size 1280 (the size of the camera images), batch 8, early stop after 30 epochs without improvement, colour (HSV) augmentation off. `--epochs`, `--imgsz` and `--batch` exist for quick tests only; never change them between the RGB and RGB-D model. Each run's dataset and image size are recorded in `run_info.json`, and `evaluate`, `test` and `detect` reuse them automatically.

## Evaluating and testing a trained model

Do this before trying the model live.

**Scores** on the held-out `test` split (and `test_white` if it exists):

```
uv run python -m vision_intelligence evaluate                                   # all runs
uv run python -m vision_intelligence evaluate --dataset <dataset_id>            # all models of one dataset
uv run python -m vision_intelligence evaluate --runs <dataset_id>/rgb_yolo11n-obb_s0
uv run python -m vision_intelligence evaluate --runs ... --data other/data.yaml # test on a different dataset
```

Prints a table (mAP50, mAP50-95, precision / recall / AP50 for every class, inference time per image), writes `model/<dataset_id>/comparison.csv`, and saves Ultralytics plots (confusion matrix, PR curves) to `model/<dataset_id>/<run>/eval_test/`. With several seeds it also prints the mean and standard deviation. Filter recall on `test_white` is the primary metric of the RGB vs RGB-D comparison.

**Visual test**: predictions drawn on the test images next to the ground-truth labels:

```
uv run python -m vision_intelligence test                                     # latest trained run, its test split
uv run python -m vision_intelligence test --run <dataset_id>/<run>            # a specific run
uv run python -m vision_intelligence test --run <dataset_id>/<run> --source path/to/new/images --conf 0.3
```

Output in `model/<dataset_id>/<run>/test_<folder>/`: one image per input (orange = filter, magenta = frame, cyan = Air_Filter, green = ground truth, red = missed), `overview.png` with all images as numbered tiles, `results.csv` (per image: correct, missed and false detections, `OK` or `CHECK`) and `predictions.json` (class, confidence, corners, approximate 3D centre, yaw). A prediction counts as correct if it has the right class and overlaps the ground truth by IoU ≥ 0.5 (`--iou`).

## Live detection test

Runs a trained model on the camera stream (the command prints which model it uses):

```
uv run python -m vision_intelligence detect                                   # latest trained run
uv run python -m vision_intelligence detect --run <dataset_id>/<run>          # a specific run
uv run python -m vision_intelligence detect --conf 0.3                        # also show less confident detections
uv run python -m vision_intelligence detect --stretch                         # only if the training images were stretched to a square
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
