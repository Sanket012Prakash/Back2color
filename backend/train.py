"""Train the Lab autoencoder on images in the Train folder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.transform import resize
from tensorflow.keras.optimizers import Adam

from autoencoder import build_colorizer

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "Train"
WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "colorizer.keras"
SIZE = 128
SUFFIXES = {".jpg", ".jpeg", ".png", ".jfif", ".bmp"}


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    for path in sorted(TRAIN_DIR.iterdir()):
        if path.suffix.lower() not in SUFFIXES:
            continue
        try:
            with Image.open(path) as img:
                rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        except Exception:
            continue
        rgb = resize(rgb, (SIZE, SIZE), anti_aliasing=True, preserve_range=True)
        lab = rgb2lab(rgb)
        X.append(lab[:, :, 0])
        Y.append(lab[:, :, 1:] / 128.0)
    if not X:
        raise RuntimeError(f"No training images found in {TRAIN_DIR}")
    x = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(Y, dtype=np.float32)
    return x, y


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print("Loading training images…")
    x, y = load_dataset()
    print(f"Loaded {len(x)} images with shape {x.shape}")
    model = build_colorizer(SIZE)
    model.compile(optimizer=Adam(1e-4), loss="mse")
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.fit(x, y, validation_split=0.1, epochs=epochs, batch_size=4, verbose=1)
    model.save(WEIGHTS_PATH)
    print(f"Saved {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
