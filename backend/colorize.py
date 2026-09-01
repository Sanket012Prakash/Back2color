"""Zhang ECCV 2016 colorization (ImageNet-trained ONNX) for arbitrary photos."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "models"
ONNX_PATH = MODELS_DIR / "colorizer.onnx"
ONNX_URL = "https://storage.googleapis.com/ailia-models/colorization/colorizer.onnx"
TRAIN_DIR = ROOT / "Train"
MAX_SIDE = 640
ONNX_INPUT = 256
MIN_ONNX_BYTES = 80_000_000

_net = None
_net_error: str | None = None


class ColorizeError(RuntimeError):
    """Raised when the model cannot colorize an image."""


def _download_onnx() -> None:
    if ONNX_PATH.exists() and ONNX_PATH.stat().st_size >= MIN_ONNX_BYTES:
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    part = ONNX_PATH.with_suffix(".onnx.part")
    print(f"Downloading colorization model to {ONNX_PATH} …")
    urllib.request.urlretrieve(ONNX_URL, part)
    if part.stat().st_size < MIN_ONNX_BYTES:
        part.unlink(missing_ok=True)
        raise ColorizeError("Downloaded colorization model looks incomplete.")
    part.replace(ONNX_PATH)


def _load_net():
    global _net, _net_error
    if _net is not None:
        return _net
    try:
        _download_onnx()
        net = cv2.dnn.readNetFromONNX(str(ONNX_PATH), cv2.dnn.ENGINE_CLASSIC)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        dummy = np.zeros((1, 1, ONNX_INPUT, ONNX_INPUT), dtype=np.float32)
        net.setInput(dummy)
        net.forward()
        _net = net
        _net_error = None
        return _net
    except Exception as exc:  # noqa: BLE001
        _net_error = str(exc)
        return None


def warmup() -> None:
    _load_net()


def engine_status() -> dict:
    net = _load_net()
    return {
        "engine": "zhang-eccv16" if net is not None else "none",
        "ready": net is not None,
        "detail": _net_error,
        "input_size": ONNX_INPUT,
    }


def _fit(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return rgb
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def _colorize_zhang(rgb: np.ndarray, net) -> np.ndarray:
    rgb_f = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    lightness = rgb2lab(rgb_f)[:, :, 0].astype(np.float32)
    height, width = lightness.shape
    l256 = cv2.resize(np.clip(lightness, 0.0, 100.0), (ONNX_INPUT, ONNX_INPUT), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(l256, 1.0, (ONNX_INPUT, ONNX_INPUT), swapRB=False, crop=False)
    net.setInput(blob)
    out = np.array(net.forward())
    ab = out[0] if out.ndim == 4 else out
    if ab.shape[0] == 2:
        ab = ab.transpose(1, 2, 0)
    ab = cv2.resize(ab.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    ab = cv2.GaussianBlur(ab, (0, 0), 0.8)
    lab = np.empty((height, width, 3), dtype=np.float64)
    lab[:, :, 0] = lightness
    lab[:, :, 1:] = np.clip(ab, -110.0, 110.0)
    rgb_out = lab2rgb(lab)
    return (np.clip(rgb_out, 0.0, 1.0) * 255.0).astype(np.uint8)


def colorize_rgb(rgb: np.ndarray) -> np.ndarray:
    work = _fit(np.ascontiguousarray(rgb))
    net = _load_net()
    if net is None:
        raise ColorizeError(_net_error or "Colorization model is not available.")
    return _colorize_zhang(work, net)


def colorize_pil(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    return Image.fromarray(colorize_rgb(rgb))


def grayscale_pil(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")
