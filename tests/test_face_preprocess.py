"""Tests for the raw-image face detection path.

Regression guard: a previous MediaPipe implementation broke silently
(``mp.solutions`` was removed in mediapipe >= 0.10), so detection always threw
and every prediction ran on the uncropped image. These tests ensure the
detector loads, never raises, and returns None (not garbage) when there's no
face — the behavior callers rely on to fall back correctly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.data.face_preprocess import _FACE_CASCADES, detect_and_align


def test_cascades_loaded() -> None:
    """At least one Haar cascade must load, or detection is dead on arrival."""
    assert len(_FACE_CASCADES) >= 1


def test_no_face_returns_none_pil() -> None:
    """A blank image has no face -> None (and must not raise)."""
    assert detect_and_align(Image.new("RGB", (200, 200), (127, 127, 127))) is None


def test_no_face_returns_none_array() -> None:
    """A random-noise ndarray has no face -> None (and must not raise)."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(180, 180, 3), dtype=np.uint8)
    assert detect_and_align(noise) is None


def test_grayscale_input_does_not_crash() -> None:
    """2-D (grayscale) input is coerced to RGB without error."""
    gray = np.full((120, 120), 100, dtype=np.uint8)
    assert detect_and_align(gray) is None  # no face, but must not raise
