from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from colorize import (
    TRAIN_DIR,
    ColorizeError,
    colorize_pil,
    engine_status,
    grayscale_pil,
    warmup,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SAMPLE_SPECS = [
    {"id": "lion", "title": "Lion", "file": TRAIN_DIR / "lion.jpeg"},
    {"id": "bridge", "title": "Golden Gate", "file": TRAIN_DIR / "golden-gate-bridge-san-francisco-2104742.jpg"},
    {"id": "elephant", "title": "Elephant", "file": TRAIN_DIR / "elephant.JPG"},
    {"id": "arc", "title": "Arc de Triomphe", "file": TRAIN_DIR / "Arc_de_Triomphe_Paris.jpg"},
    {"id": "horse", "title": "Horse", "file": TRAIN_DIR / "horse.JPG"},
    {"id": "friends", "title": "Road trip", "file": TRAIN_DIR / "friends-on-road-trip.jpg"},
]
_sample_cache: dict[str, bytes] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup()
    yield


app = FastAPI(title="Colorize B&W Pictures", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, **engine_status()}


@app.get("/api/samples")
def list_samples():
    return {
        "samples": [
            {"id": spec["id"], "title": spec["title"]}
            for spec in SAMPLE_SPECS
            if spec["file"].exists()
        ]
    }


def _jpeg(image: Image.Image, filename: str) -> Response:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


def _open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Please upload a valid image file.") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Could not read that image.") from exc


def _load_sample(sample_id: str) -> Image.Image:
    spec = next((item for item in SAMPLE_SPECS if item["id"] == sample_id), None)
    if spec is None or not spec["file"].exists():
        raise HTTPException(status_code=404, detail="Sample not found.")
    image = Image.open(spec["file"]).convert("RGB")
    image.thumbnail((512, 512))
    return image


def _colorize_or_fail(image: Image.Image) -> Image.Image:
    try:
        return colorize_pil(image)
    except ColorizeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Colorization failed: {exc}") from exc


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: str):
    cached = _sample_cache.get(sample_id)
    if cached is None:
        image = grayscale_pil(_load_sample(sample_id))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78)
        cached = buffer.getvalue()
        _sample_cache[sample_id] = cached
    return Response(content=cached, media_type="image/jpeg")


@app.post("/api/colorize")
async def colorize_upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is larger than 8 MB.")
    image = _open_image(data)
    image.thumbnail((512, 512))
    return _jpeg(_colorize_or_fail(image), "colorized.jpg")


@app.post("/api/colorize-sample/{sample_id}")
def colorize_sample(sample_id: str):
    return _jpeg(_colorize_or_fail(_load_sample(sample_id)), f"{sample_id}.jpg")


app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
