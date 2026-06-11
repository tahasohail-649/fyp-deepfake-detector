"""
FastAPI backend for Hybrid Ensemble Deepfake Detector.

Endpoints:
  POST /predict/image   — upload image → visual ensemble inference
  POST /predict/audio   — upload audio → LCNN inference
  POST /predict/video   — upload video → multimodal fusion inference
  GET  /health          — liveness check
"""

import os
import sys
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.multimodal_fusion import MultimodalDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake_api")

# ---------------------------------------------------------------------------
# Global model (loaded once at startup)
# ---------------------------------------------------------------------------
detector: Optional[MultimodalDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Loading models…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    detector = MultimodalDetector(device=device)
    logger.info(
        f"Models ready. Audio available: {detector.has_audio}"
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Deepfake Detector API",
    description="Hybrid ensemble (XceptionNet + EfficientNet-B0 + ViT) + LCNN audio anti-spoofing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if present
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _save_upload(upload: UploadFile, suffix: str) -> str:
    """Save UploadFile to a temp file; return path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[-1].lower()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "audio_model_loaded": detector.has_audio if detector else False,
        "device": str(detector.device) if detector else "not loaded",
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file.
    Returns visual ensemble prediction with per-model probabilities.
    """
    ext = _ext(file.filename)
    if ext not in ALLOWED_IMAGE:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {ALLOWED_IMAGE}",
        )

    tmp_path = _save_upload(file, suffix=ext)
    try:
        result = detector.predict_image_path(tmp_path)
    except Exception as e:
        logger.exception("Image inference error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "modality": "image",
        **result,
    }


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """
    Upload an audio file.
    Returns LCNN spoof probability.
    """
    if not detector.has_audio:
        raise HTTPException(
            status_code=503,
            detail="Audio model not loaded (checkpoint missing).",
        )

    ext = _ext(file.filename)
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {ALLOWED_AUDIO}",
        )

    tmp_path = _save_upload(file, suffix=ext)
    try:
        from src.audio_classifier import predict_audio_file
        result = predict_audio_file(tmp_path, detector.audio_model, detector.device)
    except Exception as e:
        logger.exception("Audio inference error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return {
        "filename": file.filename,
        "modality": "audio",
        **result,
    }


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    sample_frames: int = 10,
    fusion_mode: str = "weighted_avg",
):
    """
    Upload a video file.
    Extracts frames (visual) + audio track (LCNN), returns fused result.

    Query params:
      sample_frames — number of frames to sample (default 10)
      fusion_mode   — "weighted_avg" or "max_suspicion"
    """
    ext = _ext(file.filename)
    if ext not in ALLOWED_VIDEO:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {ALLOWED_VIDEO}",
        )
    if fusion_mode not in ("weighted_avg", "max_suspicion"):
        raise HTTPException(
            status_code=400,
            detail="fusion_mode must be 'weighted_avg' or 'max_suspicion'",
        )
    if not (1 <= sample_frames <= 50):
        raise HTTPException(
            status_code=400,
            detail="sample_frames must be between 1 and 50",
        )

    # Swap fusion mode if requested (without re-loading models)
    original_mode = detector.fusion_mode
    detector.fusion_mode = fusion_mode

    tmp_path = _save_upload(file, suffix=ext)
    try:
        result = detector.predict_video(tmp_path, sample_frames=sample_frames)
    except Exception as e:
        logger.exception("Video inference error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
        detector.fusion_mode = original_mode

    return {
        "filename": file.filename,
        "modality": "video",
        "sample_frames": sample_frames,
        **result,
    }


# ---------------------------------------------------------------------------
# Grad-CAM heatmap endpoint
# ---------------------------------------------------------------------------

@app.post("/predict/gradcam")
async def predict_gradcam(
    file: UploadFile = File(...),
    model_name: str = "efficientnet_b0",
):
    """
    Upload an image → returns Grad-CAM heatmap overlay as base64 PNG.

    Query param:
      model_name — "xception" | "efficientnet_b0" | "vit"
    """
    ext = _ext(file.filename)
    if ext not in ALLOWED_IMAGE:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {ALLOWED_IMAGE}",
        )
    if model_name not in detector.visual_models:
        raise HTTPException(
            status_code=400,
            detail=f"model_name must be one of {list(detector.visual_models.keys())}",
        )

    tmp_path = _save_upload(file, suffix=ext)
    try:
        from src.gradcam import run_gradcam
        from src.multimodal_fusion import VAL_TRANSFORMS
        b64_png = run_gradcam(
            tmp_path,
            detector.visual_models,
            model_name,
            detector.device,
            VAL_TRANSFORMS,
        )
    except Exception as e:
        logger.exception("Grad-CAM error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return {
        "filename":   file.filename,
        "model_name": model_name,
        "heatmap_b64": b64_png,   # data:image/png;base64,<this>
    }


# ---------------------------------------------------------------------------
# Serve frontend index at root
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Deepfake Detector API. See /docs for endpoints."}
