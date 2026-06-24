# Cat Detection — Unit 6 Final Assessment

Single-class cat detector built with YOLO26, trained on the course cat-detection dataset,
exported to ONNX, and shipped as a Docker container.

## Image for leaderboard

```bash
docker pull <your-dockerhub-user>/cat-detector:final
```

**Image:** `<your-dockerhub-user>/cat-detector:final`  
**Student:** Your Name

---

## How to reproduce

### 1. Install dependencies

```bash
pip install ultralytics onnx onnxruntime numpy pillow opencv-python-headless
```

### 2. Dataset

Download the dataset from the course Google Drive link and place it at:

```
data/DATA_CLEAN/
  images/
  labels/
```

### 3. Run the notebooks

- `m6-04-assessment.ipynb` — Week 1: inspect, split, train baseline (yolo26s, 50 ep)
- `m6-09-assessment.ipynb` — Week 2: improve, export ONNX, container instructions

### 4. Build and run the container

```bash
# Build (run from repo root)
docker build -t <your-dockerhub-user>/cat-detector:final -f container/Dockerfile .

# Test info subcommand
docker run --rm <your-dockerhub-user>/cat-detector:final info

# Test predict subcommand
mkdir -p /tmp/inp /tmp/out
cp path/to/test/images/*.jpg /tmp/inp/
docker run --rm \
  -v /tmp/inp:/data/input:ro \
  -v /tmp/out:/data/output \
  <your-dockerhub-user>/cat-detector:final predict

cat /tmp/out/predictions.csv | head

# Push
docker login
docker push <your-dockerhub-user>/cat-detector:final
```

---

## Container interface

### `info` subcommand
Prints `STUDENT.json` to stdout (valid JSON), exit code 0.

### `predict` subcommand
- **Input:**  `/data/input/` — any `.jpg/.jpeg/.png` files (recursive subdirectories OK)
- **Output:** `/data/output/predictions.csv`

CSV schema:
```
image_path,xmin,ymin,xmax,ymax,confidence,class
```

- `image_path` — relative path from `/data/input/`, forward slashes
- `xmin/ymin/xmax/ymax` — absolute pixel coordinates in original image
- `confidence` — float in [0,1]
- `class` — `cat`
- Images with no detections get one row with empty box fields

---

## Model details

| | Week-1 baseline | v2 best |
|---|---|---|
| Backbone | yolo26s | yolo26m |
| Epochs | 50 | 80 |
| Augmentation | defaults | mosaic + mixup + copy_paste + geometric |
| LR schedule | step decay | cosine (`cos_lr=True`) |
| Regularisation | none | `weight_decay=0.001`, `patience=20` |
| Export | — | ONNX opset 17, NMS-free e2e |

---

## .gitignore

```
data/
runs/cats_v1/weights/best.pt
runs/cats_v2_run*/
container/models/best.onnx
__pycache__/
*.pyc
.ipynb_checkpoints/
```
