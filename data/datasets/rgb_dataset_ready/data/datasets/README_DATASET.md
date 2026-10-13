# RGB dataset: filters and frames (YOLO OBB)

Ready to train with Ultralytics YOLO-OBB. Unzip into the repository root (`Vision-Intelligence/`)
so the files land in `data/datasets/`, then train from the repository root:

```python
from ultralytics import YOLO
YOLO("yolo11n-obb.pt").train(data="data/datasets/rgb.yaml", imgsz=1280, epochs=100, batch=8, device=0)
```

## Contents

| Path | What it is |
| --- | --- |
| `rgb/images/{train,val,test}/` | 25 colour images, 1280×720, original names from `index.csv` |
| `rgb/labels/{train,val,test}/` | YOLO OBB labels: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalized 0–1 |
| `rgb.yaml` | Dataset config (`0 = filter`, `1 = frame`, `channels: 3`) |
| `index.csv` | Your index with the `split` column filled in |
| `labels_report.csv` | One row per box: class, size in pixels, fill ratio, flags |
| `label_check/` | Every image with its boxes drawn (blue = filter, red = frame) + `overview.jpg` |

## Counts

| Split | Images | Filter boxes | Frame boxes |
| --- | --- | --- | --- |
| train | 17 | 16 | 25 |
| val | 4 | 4 | 6 |
| test | 4 | 4 | 6 |

## How the labels were made

Labels were generated from the colour images (no depth was available in the upload) and every image
was checked visually afterwards:

- **Filter:** bright, low-saturation (white) region, size-checked as a long narrow rectangle; rotated box from `cv2.minAreaRect`.
- **Frame:** dark/grey region inside the table and cardboard area, closed into one shape so the hollow frame gets its outer box.
- **Static distractors excluded:** the metal rail and bench on the left, the white board, cables, the white plate, shoes.
- **One manual correction:** image `frames_20261010_162901_20261010_162901_0007`: the frame hangs over the cardboard edge and was split by segmentation, so its 4 corners were set by hand.

The scripts are in `tools/rgb_auto_labelling/` for reference.

## Labelling rules used

- Every visible filter and frame is labelled, regardless of the source folder. The frame on the wooden table appears in every image and is labelled every time.
- Filters standing on their long side (edge-on) are labelled with a thin box.
- Filters held in a person's hand (frames session) are labelled on their visible part; `labels_report.csv` marks them `handheld=True`.
- Slivers barely visible at the image edge are not labelled (e.g. bottom edge of `frames_20261010_162901_20261010_162901_0003`).

## Split

Split by contiguous time blocks within each capture session (about 70/15/15), because there is only one
session per category. Near-identical frames therefore do not appear in both train and test.

## Known limitations

- **Very small dataset:** 25 images from 2 sessions, one camera position, one lighting setup. Treat results as a first baseline.
- **No `test_white` split:** none of these images shows a filter on a white background. Record a session with white backgrounds to test the RGB vs RGB-D question.
- **RGB-D dataset not included:** depth files were not part of the upload. Build it with Phase 2 of `METHODOLOGY.md`; the labels and splits here can be reused unchanged because depth is aligned to colour.
