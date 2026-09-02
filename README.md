# Colorization of Black and White pictures

Colorize grayscale photos with a pretrained Lab-space network (Zhang et al., ECCV 2016). The app keeps lightness **L** and predicts the color channels **ab**, then converts back to RGB.

The original notebook `Colorization_of_BW_pictures.ipynb` trains a convolutional autoencoder on the `Train/` images. The Streamlit app uses a stronger ImageNet-trained colorizer so arbitrary uploads get plausible color, not colors copied from another photo.

## Live Demo
https://back2color-juosuvh4tvhf4n9an3appu4.streamlit.app/

## Streamlit app

```text
streamlit_app.py     Streamlit UI (upload, samples, download)
backend/colorize.py  Zhang ONNX colorization
backend/models/      colorizer.onnx (downloaded on first run, ~129 MB)
Train/               sample photos
.streamlit/          theme and upload limits
requirements.txt     Python packages for Streamlit
```

### Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501). Drop a photo or click a sample. On first launch the app downloads `backend/models/colorizer.onnx` if it is missing.

## Notebook autoencoder

RGB is converted to Lab before training. The encoder uses stride-2 convolutions; the decoder upsamples and predicts two `tanh` filters for *ab*. Loss is MSE; the optimizer is Adam.

```bash
cd backend
pip install -r requirements.txt
python train.py 8
```

## Optional FastAPI UI

```bash
pip install -r backend/requirements.txt
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
