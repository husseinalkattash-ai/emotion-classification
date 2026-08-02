# Facial Emotion Classification (PyTorch)

A transfer-learning model that classifies face images into **7 emotions**
(`surprise, fear, disgust, happy, sad, angry, neutral`), plus a **web app**
(FastAPI + a small HTML/JS frontend) for uploading a photo and getting a live
prediction with visual explanations.

Full pipeline: data loading, transfer-learning model (ResNet50 / EfficientNet-B0),
training loop, evaluation, inference, Grad-CAM interpretability, and a web UI.
Runs on a local machine with a CUDA GPU (falls back to CPU).

> **Scope note:** This recognizes **facial expressions**, which are not the same
> as a person's true internal emotions. It is **NOT** a medical diagnostic tool
> and contains no logic that claims to diagnose dementia or any medical condition.

## Highlights

- **~84% test accuracy** on RAF-DB (macro-F1 ≈ 0.76) with ResNet50.
- **Class-imbalance handling** via a balanced sampler and/or class weights.
- **Optional extra training data** (e.g. AffectNet) pooled in to strengthen rare
  classes — validation/test stay RAF-DB-only so metrics remain comparable.
- **Test-time augmentation (TTA)** for a small, free accuracy boost.
- **Web app** with face **detection + eye-alignment** (matches training framing),
  a **Grad-CAM heatmap**, the **analyzed face crop**, and an **uncertainty flag**
  for ambiguous predictions (e.g. fear vs. surprise).

---

## 1. Requirements & setup on a new machine

### 1.1 System requirements

- **OS:** Windows, Linux, or macOS.
- **Python:** 3.10 or newer.
- **git** (to clone the repo).
- **GPU (optional):** an NVIDIA GPU with a current CUDA driver makes training/inference much faster. Without one, everything still runs on CPU (slower).
- **Disk:** roughly 3–5 GB for the Python dependencies (PyTorch is the large one).
- **Internet:** needed once, to install packages and (if missing) to auto-download the small face-detector model.

### 1.2 Get the code

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 1.3 Create a virtual environment (recommended)

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 1.4 Install PyTorch first (pick ONE)

Install PyTorch matching the machine, then the rest. Get the exact command for your CUDA from <https://pytorch.org/get-started/locally/>.

```bash
# GPU (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# OR CPU-only (no NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 1.5 Install the remaining dependencies

```bash
pip install -r requirements.txt
```

This installs everything the project uses: `numpy`, `pandas`, `pillow`,
`opencv-python-headless`, `mediapipe` (face detection/alignment), `scikit-learn`,
`matplotlib`, `seaborn`, `pyyaml`, `tqdm`, `tensorboard`, `grad-cam`, `pytest`,
and the web-app stack (`fastapi`, `uvicorn[standard]`, `python-multipart`).

### 1.6 Verify the install

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
pytest -q          # the test suite should pass (no dataset needed)
```

### 1.7 IMPORTANT — the trained model is NOT in the repo

`data/` (datasets) and `outputs/` (checkpoints, logs) are **git-ignored**, so a
fresh clone contains **code only** — no trained model. To run the web app or make
predictions you need a checkpoint at `outputs/best_model.pth`. Two options:

- **Copy the checkpoint** `outputs/best_model.pth` from the machine where you
  trained, into the same path here. (This is all the web app needs — no dataset
  required to run it.)
- **Or train from scratch** on this machine (Sections 2–3), which does require
  downloading the dataset.

### 1.8 The face-detector model

Face detection/alignment uses a small BlazeFace model at
`assets/blaze_face_short_range.tflite`. It ships with the repo; if it is ever
missing, the code auto-downloads it on first use (needs internet that once).

---

## 2. Obtain & place the dataset (RAF-DB)

The loader supports **two on-disk layouts**, selected by `data.format`.

### Option A — Kaggle mirror (`data.format: imagefolder`, the default)

```bash
pip install kaggle                       # place your API token at ~/.kaggle/kaggle.json
kaggle datasets download -d shuvoalok/raf-db-dataset -p data/rafdb --unzip
```

Expected layout (class folder name = RAF-DB 1-indexed id, or an emotion name):

```
data/rafdb/
+-- DATASET/
    +-- train/  1/ 2/ 3/ 4/ 5/ 6/ 7/    # 1=surprise ... 7=neutral
    +-- test/   1/ 2/ 3/ 4/ 5/ 6/ 7/
```

If your archive extracts differently, adjust `data.root`, `data.train_dir`,
`data.test_dir` in `configs/default.yaml`.

### Option B — Official RAF-DB (`data.format: rafdb_official`)

License-gated — request access at <http://www.whdeng.cn/RAF/model1.html>, place
the *basic* subset under `data/rafdb/basic/…`, and set `data.format: rafdb_official`.

### Verify placement (either layout)

```bash
python scripts/download_data.py --config configs/default.yaml
```

### Label mapping (critical)

RAF-DB uses an official **1-indexed** labeling, mapped to the model's
**0-indexed** output order via an explicit `RAFDB_LABEL_MAP` in
`src/data/dataset.py` — never assumed alphabetically.

| RAF-DB ID | 1 Surprise | 2 Fear | 3 Disgust | 4 Happy | 5 Sad | 6 Angry | 7 Neutral |
|-----------|-----------|--------|-----------|---------|-------|---------|-----------|
| Model idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 |

### (Optional) Extra training data — e.g. AffectNet

To strengthen the rare classes (fear/disgust), you can pool an auxiliary dataset
into the **train split only** (val/test stay RAF-DB). Download AffectNet
(license-gated / Kaggle mirrors exist), then enable it in `configs/default.yaml`:

```yaml
data:
  extra_train_datasets:
    - name: affectnet
      root: "data/affectnet/archive (3)"
      train_dir: Train
      class_alias:            # map AffectNet's classes onto ours
        anger: angry
        contempt: drop        # 8th class excluded (RAF-DB has 7)
```

---

## 3. Train / evaluate / predict (CLI)

```bash
# Train (best checkpoint -> outputs/best_model.pth, selected by val macro-F1)
python scripts/train.py --config configs/default.yaml

# Evaluate on the test split: classification report + confusion matrix in outputs/
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pth --tta

# Evaluate + Grad-CAM overlays into outputs/gradcam/
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pth --gradcam

# Predict on a new image or folder (writes outputs/predictions.csv)
python scripts/predict.py --checkpoint outputs/best_model.pth --input path/to/image_or_folder

# TensorBoard
tensorboard --logdir outputs --host 0.0.0.0 --port 6006
```

Run the tests:

```bash
pytest
```

---

## 4. Run the web app

Serves the API and the frontend together. Loads `outputs/best_model.pth` at
startup, so train (or drop a checkpoint there) first.

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000> and upload a face photo. The result panel shows
the predicted emotion, per-class probabilities, the **analyzed (aligned) face
crop**, a **Grad-CAM attention heatmap**, and a **low-confidence warning** when
the top two classes are close.

**API:** `GET /api/health` and `POST /api/predict` (multipart `file`; optional
query params `tta=true|false`, `explain=true|false`).

---

## 5. Configuration

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml).
No paths are hardcoded; everything is read from the config. Key options:

- `model.backbone`: `resnet50` (default) or `efficientnet_b0`; `model.freeze_ratio`.
- `train.use_class_weights` / `train.balanced_sampler`: imbalance handling.
- `train.label_smoothing`, `train.backbone_lr_mult` (discriminative fine-tuning).
- `train.amp`, `train.batch_size`, `train.samples_per_epoch`.
- `augment.*`: flip, rotation, color jitter, random-resized-crop, random-erasing.
- `data.format`, `data.val_split`, `data.extra_train_datasets`.

---

## 6. Project structure

```
emotion-classification/
+-- app/                        # web app
|   +-- main.py                 # FastAPI: /api/health, /api/predict + static serving
|   +-- static/                 # index.html, styles.css, app.js
+-- assets/                     # blaze_face_short_range.tflite (face detector)
+-- configs/default.yaml        # all hyperparameters
+-- data/                       # (git-ignored) datasets go here
+-- outputs/                    # (git-ignored) checkpoints, logs, gradcam, plots
+-- src/
|   +-- config.py               # load & validate YAML config
|   +-- data/
|   |   +-- dataset.py          # RAF-DB Dataset, transforms, RAFDB_LABEL_MAP, extra datasets
|   |   +-- face_preprocess.py  # face detection + eye-alignment (inference)
|   +-- models/classifier.py    # EmotionClassifier (backbone + head)
|   +-- engine/
|   |   +-- train.py            # training loop (AMP, cosine, early stopping, sampler)
|   |   +-- evaluate.py         # metrics, confusion matrix, TTA
|   |   +-- utils.py            # seed, checkpointing, meters
|   +-- interpret/gradcam.py    # Grad-CAM (batch + single in-memory overlay)
|   +-- inference.py            # predict on a new image/folder -> CSV
+-- scripts/                    # thin CLI entry points (download_data/train/evaluate/predict)
+-- tests/                      # pytest (synthetic fixtures, no real data needed)
+-- requirements.txt
+-- README.md
```

**Design notes**

- Logic lives in `src/`; `scripts/` are thin `argparse` entry points; `app/` is the web layer.
- RAF-DB ships **pre-aligned** crops (used directly for training). At **inference**
  on raw photos, `face_preprocess.py` detects the eyes and applies a similarity
  transform onto a template *measured from RAF-DB crops*, so live images are framed
  like the training data.
- Best checkpoint is selected by **validation macro-F1** (not accuracy).
- Data and checkpoints are never committed (see `.gitignore`).

> **Note on real-world data:** `src/data/dataset.py` has a `TODO` marking where a
> **subject-level split** should be used when training on real application images,
> to avoid identity leakage between train and val/test.
