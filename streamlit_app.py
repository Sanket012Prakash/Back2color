from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from colorize import (  # noqa: E402
    TRAIN_DIR,
    ColorizeError,
    colorize_pil,
    engine_status,
    grayscale_pil,
    warmup,
)

SAMPLES = [
    {"id": "lion", "title": "Lion", "file": TRAIN_DIR / "lion.jpeg"},
    {"id": "bridge", "title": "Golden Gate", "file": TRAIN_DIR / "golden-gate-bridge-san-francisco-2104742.jpg"},
    {"id": "elephant", "title": "Elephant", "file": TRAIN_DIR / "elephant.JPG"},
    {"id": "arc", "title": "Arc de Triomphe", "file": TRAIN_DIR / "Arc_de_Triomphe_Paris.jpg"},
    {"id": "horse", "title": "Horse", "file": TRAIN_DIR / "horse.JPG"},
    {"id": "friends", "title": "Road trip", "file": TRAIN_DIR / "friends-on-road-trip.jpg"},
]


def _available_samples() -> list[dict]:
    return [item for item in SAMPLES if item["file"].exists()]


@st.cache_resource(show_spinner="Loading colorization model…")
def _engine():
    warmup()
    return engine_status()


@st.cache_data(show_spinner=False)
def _grayscale_jpeg(data: bytes) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((640, 640))
    gray = grayscale_pil(image)
    buffer = io.BytesIO()
    gray.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def _sample_preview(path: str) -> bytes:
    image = Image.open(path).convert("RGB")
    image.thumbnail((240, 240))
    gray = grayscale_pil(image)
    buffer = io.BytesIO()
    gray.save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def _colorize_jpeg(data: bytes) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((640, 640))
    colorized = colorize_pil(image)
    buffer = io.BytesIO()
    colorized.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _set_source(data: bytes, label: str) -> None:
    st.session_state.source_bytes = data
    st.session_state.source_label = label


st.set_page_config(
    page_title="Chroma — Colorize B&W Pictures",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; max-width: 1180px; }
      h1 { font-weight: 650; letter-spacing: -0.03em; }
      div[data-testid="stCaptionContainer"] { color: #b7a894; }
    </style>
    """,
    unsafe_allow_html=True,
)

status = _engine()
status_line = (
    "Model ready · Zhang ECCV 2016"
    if status.get("ready")
    else f"Model not ready: {status.get('detail') or 'unknown error'}"
)

top_left, top_right = st.columns([2, 1])
with top_left:
    st.caption("CHROMA")
    st.title("Bring black & white photographs back to life.")
with top_right:
    st.caption(status_line)

st.write(
    "Upload a photo or pick a sample. The app converts it to grayscale, then a "
    "pretrained Lab-space network predicts color for that picture."
)

if "source_bytes" not in st.session_state:
    st.session_state.source_bytes = None
    st.session_state.source_label = ""

uploaded = st.file_uploader(
    "Drop a photo",
    type=["jpg", "jpeg", "png", "webp", "bmp", "jfif"],
    help="JPG, PNG, WEBP · up to 8 MB",
)
if uploaded is not None:
    if uploaded.size > 8 * 1024 * 1024:
        st.error("Image is larger than 8 MB.")
    else:
        upload_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("upload_id") != upload_id:
            st.session_state.upload_id = upload_id
            _set_source(uploaded.getvalue(), uploaded.name)

samples = _available_samples()
if samples:
    st.subheader("Or try a sample")
    columns = st.columns(len(samples))
    for column, sample in zip(columns, samples):
        with column:
            st.image(
                _sample_preview(str(sample["file"])),
                caption=sample["title"],
                width="stretch",
            )
            if st.button(sample["title"], key=f"sample-{sample['id']}", width="stretch"):
                _set_source(sample["file"].read_bytes(), sample["title"])

if not status.get("ready"):
    st.error("The colorization model could not be loaded. Check your network and try again.")
    st.stop()

if st.session_state.source_bytes:
    try:
        with st.spinner("Painting in color…"):
            gray_bytes = _grayscale_jpeg(st.session_state.source_bytes)
            color_bytes = _colorize_jpeg(st.session_state.source_bytes)
    except ColorizeError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Colorization failed: {exc}")
    else:
        left, right = st.columns(2)
        with left:
            st.image(gray_bytes, caption="Original", width="stretch")
        with right:
            st.image(color_bytes, caption="Colorized", width="stretch")
        st.download_button(
            "Download colorized JPEG",
            data=color_bytes,
            file_name="colorized.jpg",
            mime="image/jpeg",
        )
        if st.session_state.source_label:
            st.caption(f"Source: {st.session_state.source_label}")
else:
    st.info("Choose a sample or upload a photo to colorize it.")
