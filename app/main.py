"""FastAPI backend for emotion classification inference.

Loads the trained model checkpoint once at startup, exposes a POST
endpoint for image-based emotion prediction, and serves the static
HTML/CSS/JS frontend.
"""
from __future__ import annotations

import base64
import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure the project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_transforms  # noqa: E402
from src.data.face_preprocess import detect_and_align  # noqa: E402
from src.inference import load_model_from_checkpoint  # noqa: E402
from src.interpret.gradcam import compute_gradcam_overlay  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Global state populated at startup ──────────────────────────────────
_model: Any = None
_classes: list[str] = []
_transform: Any = None
_device: torch.device = torch.device("cpu")

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "best_model.pth"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Uncertainty thresholds: a prediction is "confident" only if the top class is
# clear enough AND well ahead of the runner-up. Otherwise we flag it ambiguous.
MIN_CONFIDENCE = 0.50  # top-1 probability floor
MIN_MARGIN = 0.15      # top-1 minus top-2 probability


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model into memory once when the server starts."""
    global _model, _classes, _transform, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint from %s on %s …", CHECKPOINT_PATH, _device)
    _model, _classes, image_size = load_model_from_checkpoint(CHECKPOINT_PATH, _device)
    _transform = build_transforms(image_size, train=False)
    logger.info("Model ready — %d classes: %s", len(_classes), _classes)

    yield  # Application runs

    logger.info("Shutting down — releasing model.")


app = FastAPI(
    title="Emotion Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """Prevent the browser from serving stale HTML/JS/CSS during development."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ── API Endpoints ──────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Health check — confirms the model is loaded and ready."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": str(_device),
        "classes": _classes,
    }


def _to_data_uri(img: "Image.Image | np.ndarray", max_side: int = 256) -> str:
    """Encode a PIL image or RGB array as a base64 PNG data URI (downscaled)."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/predict")
async def predict_emotion(
    file: UploadFile = File(...), tta: bool = True, explain: bool = True
) -> JSONResponse:
    """Accept an image upload and return emotion predictions.

    Args:
        file: The uploaded image.
        tta: Horizontal-flip test-time augmentation (default on). Averages
            softmax over the image and its mirror to match the best evaluated
            model (~+0.6% accuracy). Pass ``?tta=false`` to disable.
        explain: If on (default), also return the analyzed face crop and a
            Grad-CAM attention heatmap as data URIs. Pass ``?explain=false``
            to skip (slightly faster).
    """
    # ── Validate the upload ────────────────────────────────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    # ── Open and preprocess ────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    face_detected = True
    try:
        aligned = detect_and_align(image)
        if aligned is not None:
            image = aligned
        else:
            face_detected = False
            logger.info("No face detected in upload; using whole image.")
    except Exception as exc:
        face_detected = False
        logger.warning("Face detection failed (%s); using whole image.", exc)

    # ── Run inference (with optional horizontal-flip TTA) ──────────────
    tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        prob = F.softmax(_model(tensor), dim=1)
        if tta:
            prob = (prob + F.softmax(_model(torch.flip(tensor, dims=[3])), dim=1)) / 2
        probs = prob[0].cpu()

    pred_idx = int(probs.argmax())
    probabilities = {_classes[i]: round(float(probs[i]), 4) for i in range(len(_classes))}

    # ── Certainty assessment (top-1 vs runner-up) ──────────────────────
    order = probs.argsort(descending=True)
    top1, top2 = int(order[0]), int(order[1])
    margin = float(probs[top1] - probs[top2])
    confident = bool(probs[top1] >= MIN_CONFIDENCE and margin >= MIN_MARGIN)

    response: Dict[str, Any] = {
        "predicted": _classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": probabilities,
        "face_detected": face_detected,
        "certainty": {
            "confident": confident,
            "margin": round(margin, 4),
            "runner_up": _classes[top2],
            "runner_up_prob": round(float(probs[top2]), 4),
        },
    }

    # ── Explainability: the analyzed crop + Grad-CAM heatmap ───────────
    if explain:
        try:
            response["crop_image"] = _to_data_uri(image)
            overlay = compute_gradcam_overlay(_model, tensor, _model.target_layer, pred_idx)
            response["gradcam_image"] = _to_data_uri(overlay)
        except Exception as exc:  # never let explainability break a prediction
            logger.warning("Grad-CAM/explain failed (%s); returning prediction only.", exc)

    return JSONResponse(content=response)


# ── Serve the static frontend ─────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
