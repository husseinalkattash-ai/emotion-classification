"""Face detection + alignment for the RAW-image inference path only.

RAF-DB training images are already eye-aligned, so this module is used solely
when predicting on new, unaligned images. To match the training distribution as
closely as possible we perform the SAME kind of alignment RAF-DB used: detect
the eyes and apply a similarity transform that maps them onto a fixed template.

The template below was measured empirically by running the face detector over
~600 RAF-DB aligned crops (mean normalized eye positions), so aligned inference
images are framed just like the crops the model trained on.

Detector: MediaPipe Tasks BlazeFace (short-range), which returns eye keypoints
and works with mediapipe >= 0.10. Falls back to an OpenCV Haar bounding-box crop
(no rotation) if MediaPipe or the model file is unavailable.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── RAF-DB alignment template (measured; normalized to [0,1]) ──────────
# subject's right eye (left side of image), subject's left eye (right side).
_TEMPLATE_RIGHT_EYE = (0.2931, 0.3369)
_TEMPLATE_LEFT_EYE = (0.7005, 0.3343)
_ALIGN_SIZE = 224  # output square size for aligned crops

# ── MediaPipe BlazeFace model (auto-downloaded once) ───────────────────
_MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "blaze_face_short_range.tflite"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_detector = None  # lazy singleton

# ── OpenCV Haar fallback cascades ──────────────────────────────────────
_CASCADE_NAMES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
]
_FACE_CASCADES = []
for _name in _CASCADE_NAMES:
    _c = cv2.CascadeClassifier(cv2.data.haarcascades + _name)
    if not _c.empty():
        _FACE_CASCADES.append(_c)
if not _FACE_CASCADES:  # pragma: no cover
    logger.warning("Could not load any Haar cascade; Haar fallback disabled.")


def _ensure_model() -> bool:
    """Ensure the BlazeFace model file exists, downloading it once if missing."""
    if _MODEL_PATH.is_file():
        return True
    try:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading BlazeFace model to %s …", _MODEL_PATH)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)  # noqa: S310
        return _MODEL_PATH.is_file()
    except Exception as exc:  # offline / blocked
        logger.warning("Could not download BlazeFace model (%s); using Haar fallback.", exc)
        return False


def _get_detector():
    """Lazily create the MediaPipe FaceDetector singleton (or None if unavailable)."""
    global _detector
    if _detector is not None:
        return _detector
    if not _ensure_model():
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        opts = vision.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            min_detection_confidence=0.5,
        )
        _detector = vision.FaceDetector.create_from_options(opts)
        logger.info("MediaPipe FaceDetector ready (landmark alignment enabled).")
    except Exception as exc:  # pragma: no cover
        logger.warning("MediaPipe Tasks unavailable (%s); using Haar fallback.", exc)
        _detector = None
    return _detector


def _to_rgb_array(image: "Image.Image | np.ndarray") -> np.ndarray:
    """Coerce a PIL image or array to an HxWx3 uint8 RGB array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3]


def _detect_eyes(rgb: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return ((right_eye_xy), (left_eye_xy)) in pixels, or None."""
    detector = _get_detector()
    if detector is None:
        return None
    try:
        import mediapipe as mp

        h, w = rgb.shape[:2]
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    except Exception as exc:  # pragma: no cover
        logger.warning("Face detection error (%s).", exc)
        return None
    if not result.detections:
        return None
    # Largest detection (by bbox area).
    det = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    kp = det.keypoints  # [right eye, left eye, nose, mouth, right ear, left ear]
    if len(kp) < 2:
        return None
    right_eye = (kp[0].x * w, kp[0].y * h)
    left_eye = (kp[1].x * w, kp[1].y * h)
    return right_eye, left_eye


def _align_to_template(rgb: np.ndarray, right_eye, left_eye, size: int = _ALIGN_SIZE) -> Optional[Image.Image]:
    """Similarity-transform the image so the eyes land on the RAF-DB template."""
    src = np.array([right_eye, left_eye], dtype=np.float32)
    dst = np.array(
        [
            (_TEMPLATE_RIGHT_EYE[0] * size, _TEMPLATE_RIGHT_EYE[1] * size),
            (_TEMPLATE_LEFT_EYE[0] * size, _TEMPLATE_LEFT_EYE[1] * size),
        ],
        dtype=np.float32,
    )
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None
    aligned = cv2.warpAffine(rgb, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return Image.fromarray(aligned)


def _crop_with_haar(rgb: np.ndarray, margin: float, min_neighbors: int) -> Optional[Image.Image]:
    """Fallback: axis-aligned Haar bounding-box crop (no rotation alignment)."""
    if not _FACE_CASCADES:
        return None
    h, w = rgb.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    faces = []
    for cascade in _FACE_CASCADES:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(40, 40))
        if len(faces) > 0:
            break
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(margin * fw), int(margin * fh)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
    if x1 <= x0 or y1 <= y0:
        return None
    return Image.fromarray(rgb[y0:y1, x0:x1])


def detect_and_align(
    image: "Image.Image | np.ndarray",
    margin: float = 0.25,
    min_neighbors: int = 5,
) -> Optional[Image.Image]:
    """Detect a face and return an eye-aligned RGB crop matching RAF-DB framing.

    Primary path: MediaPipe BlazeFace eyes -> similarity transform onto the
    measured RAF-DB template (rotation + scale + translation), output
    ``_ALIGN_SIZE`` square. Fallback: OpenCV Haar bounding-box crop (no
    rotation). Returns ``None`` if no face is found by either method.

    Args:
        image: Input PIL image or HxWx3 array.
        margin: Haar-fallback padding fraction around the detected box.
        min_neighbors: Haar-fallback ``minNeighbors``.

    Returns:
        An aligned/cropped ``PIL.Image``, or ``None``.
    """
    rgb = _to_rgb_array(image)

    eyes = _detect_eyes(rgb)
    if eyes is not None:
        aligned = _align_to_template(rgb, eyes[0], eyes[1])
        if aligned is not None:
            return aligned

    # Fallback to a plain Haar crop.
    crop = _crop_with_haar(rgb, margin, min_neighbors)
    if crop is None:
        logger.warning("No face detected in image.")
    return crop
