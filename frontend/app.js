const API = (window.location.port === "8000" || window.location.port === "")
  ? ""
  : "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const colorizeBtn = document.getElementById("colorizeBtn");
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");
const hint = document.getElementById("hint");
const stage = document.getElementById("stage");
const beforeImage = document.getElementById("beforeImage");
const afterImage = document.getElementById("afterImage");
const sampleGrid = document.getElementById("sampleGrid");
const busy = document.getElementById("busy");
const engineStatus = document.getElementById("engineStatus");

let selectedFile = null;
let selectedSampleId = null;
let resultUrl = null;
let colorizing = false;

function setHint(text, isError = false) {
  hint.textContent = text;
  hint.classList.toggle("error", isError);
}

function showBusy(on) {
  busy.hidden = !on;
}

function shrinkFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, 512 / Math.max(img.naturalWidth, img.naturalHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
      const ctx = canvas.getContext("2d");
      ctx.filter = "grayscale(1)";
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Could not read image."));
          return;
        }
        resolve({ blob, preview: canvas.toDataURL("image/jpeg", 0.85) });
      }, "image/jpeg", 0.85);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read that image file."));
    };
    img.src = url;
  });
}

function clearResult() {
  if (resultUrl) {
    URL.revokeObjectURL(resultUrl);
    resultUrl = null;
  }
  downloadBtn.disabled = true;
}

async function selectImage(file, nextPreview = null, sampleId = null) {
  selectedFile = file;
  selectedSampleId = sampleId;
  clearResult();
  colorizeBtn.disabled = false;
  resetBtn.disabled = false;
  setHint(sampleId ? `Colorizing “${sampleId}”…` : "Colorizing your photo…");
  document.querySelectorAll(".sample").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.id === sampleId);
  });
  if (!nextPreview) {
    const gray = await shrinkFile(file);
    selectedFile = new File([gray.blob], file.name || "photo.jpg", { type: "image/jpeg" });
    nextPreview = gray.preview;
  }
  beforeImage.src = nextPreview;
  afterImage.src = nextPreview;
  stage.hidden = false;
  await colorize();
}

async function colorize() {
  if (colorizing || (!selectedFile && !selectedSampleId)) return;
  colorizing = true;
  showBusy(true);
  setHint("Colorizing…");
  try {
    let response;
    if (selectedSampleId) {
      response = await fetch(`${API}/api/colorize-sample/${selectedSampleId}`, { method: "POST" });
    } else {
      const body = new FormData();
      body.append("file", selectedFile, selectedFile.name || "photo.jpg");
      response = await fetch(`${API}/api/colorize`, { method: "POST", body });
    }
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Colorization failed." }));
      const detail = Array.isArray(err.detail) ? err.detail[0]?.msg : err.detail;
      throw new Error(detail || "Colorization failed.");
    }
    const blob = await response.blob();
    clearResult();
    resultUrl = URL.createObjectURL(blob);
    afterImage.src = resultUrl;
    downloadBtn.disabled = false;
    setHint("Done. Grayscale is on the left, color on the right.");
  } catch (error) {
    setHint(error.message, true);
  } finally {
    colorizing = false;
    showBusy(false);
  }
}

function resetWorkspace() {
  selectedFile = null;
  selectedSampleId = null;
  clearResult();
  colorizeBtn.disabled = true;
  resetBtn.disabled = true;
  fileInput.value = "";
  stage.hidden = true;
  beforeImage.removeAttribute("src");
  afterImage.removeAttribute("src");
  document.querySelectorAll(".sample").forEach((btn) => btn.classList.remove("active"));
  setHint("Click a sample below or drop a photo to begin.");
}

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    await selectImage(file);
  } catch (error) {
    setHint(error.message, true);
  }
});

["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
  });
});

dropzone.addEventListener("drop", async (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  try {
    await selectImage(file);
  } catch (error) {
    setHint(error.message, true);
  }
});

colorizeBtn.addEventListener("click", colorize);
resetBtn.addEventListener("click", resetWorkspace);
downloadBtn.addEventListener("click", () => {
  if (!resultUrl) return;
  const link = document.createElement("a");
  link.href = resultUrl;
  link.download = "colorized.jpg";
  link.click();
});

async function loadHealth() {
  try {
    const response = await fetch(`${API}/api/health`);
    const data = await response.json();
    engineStatus.classList.remove("ready", "warn", "bad");
    if (data.ready) {
      engineStatus.textContent = data.engine === "zhang-eccv16"
        ? "Backend ready · Zhang colorization"
        : "Backend ready";
      engineStatus.classList.add("ready");
      return true;
    }
    engineStatus.textContent = data.detail || "Model not loaded";
    engineStatus.classList.add("warn");
    return false;
  } catch {
    engineStatus.textContent = "Backend unreachable — start it on port 8000";
    engineStatus.classList.remove("ready", "warn");
    engineStatus.classList.add("bad");
    return false;
  }
}

async function loadSamples() {
  try {
    const response = await fetch(`${API}/api/samples`);
    const data = await response.json();
    sampleGrid.innerHTML = "";
    if (!data.samples?.length) {
      sampleGrid.innerHTML = "<p class='hint'>No sample photos found in the Train folder.</p>";
      return;
    }
    for (const sample of data.samples) {
      const button = document.createElement("button");
      button.className = "sample";
      button.type = "button";
      button.dataset.id = sample.id;
      button.innerHTML = `<img alt="${sample.title}" src="${API}/api/samples/${sample.id}" /><span>${sample.title}</span>`;
      button.addEventListener("click", async () => {
        try {
          const imageResponse = await fetch(`${API}/api/samples/${sample.id}`);
          const blob = await imageResponse.blob();
          const file = new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" });
          await selectImage(file, `${API}/api/samples/${sample.id}`, sample.id);
        } catch (error) {
          setHint(error.message, true);
        }
      });
      sampleGrid.appendChild(button);
    }
  } catch {
    sampleGrid.innerHTML = "<p class='hint error'>Could not load samples. Is the backend running?</p>";
  }
}

async function boot() {
  const ready = await loadHealth();
  await loadSamples();
  if (!ready) {
    setTimeout(boot, 2500);
  }
}

boot();
