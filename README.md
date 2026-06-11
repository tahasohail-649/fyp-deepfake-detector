# Deepfake Detector

> Hybrid Ensemble Deepfake Detector — Final Year Project, Department of Computer Science, Hazara University Mansehra (2026)

A multimodal deepfake detection system combining visual ensemble learning with audio anti-spoofing, served via a FastAPI backend and interactive web UI.

---

## Architecture

### Visual Pipeline
| Model | Backbone | Params | Val AUC |
|-------|----------|--------|---------|
| XceptionNet | `xception` (timm) | ~22M | — |
| EfficientNet-B0 | `efficientnet_b0` (timm) | ~5.3M | — |
| ViT-Small | `vit_small_patch16_224` (timm) | ~22M | — |
| **Visual Ensemble** | Weighted average | — | **0.9903** |

### Audio Pipeline
| Model | Input | Val AUC (dev) | EER (eval) |
|-------|-------|---------------|------------|
| LCNN | Log-mel (80×300) | **1.0000** | 33.26% |

### Multimodal Fusion
```
visual_prob × 0.6 + audio_prob × 0.4  →  ensemble_prob  →  FAKE / REAL
```
Fallback to visual-only when no audio track present.

---

## Project Structure

```
fyp-deepfake-detector/
├── api/
│   └── main.py              # FastAPI backend (image / audio / video / gradcam endpoints)
├── src/
│   ├── dataset.py           # Visual dataset loader (FaceForensics++, Celeb-DF, DFDC)
│   ├── train.py             # EfficientNet-B0 trainer
│   ├── train_xception.py    # XceptionNet trainer
│   ├── train_vit.py         # ViT-Small trainer
│   ├── ensemble.py          # Visual ensemble inference
│   ├── evaluate_ensemble.py # Visual ensemble evaluation
│   ├── audio_dataset.py     # ASVspoof2019 LA dataset loader
│   ├── train_audio.py       # LCNN trainer
│   ├── audio_classifier.py  # LCNN inference
│   ├── evaluate_audio.py    # Audio evaluation (EER, AUC)
│   ├── multimodal_fusion.py # MultimodalDetector — combines visual + audio
│   └── gradcam.py           # Grad-CAM explainability
├── frontend/
│   └── index.html           # Web UI (drag-drop, animated verdict, history, heatmap)
├── checkpoints/             # Model weights (not tracked in git)
├── logs/                    # Training/eval logs and JSON results
├── requirements-api.txt     # FastAPI dependencies
└── requirements-mlops.txt   # Training dependencies
```

---

## Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| FaceForensics++ | Visual — face manipulation | ~1000 videos, 4 methods |
| Celeb-DF v2 | Visual — celebrity deepfakes | ~6000 clips |
| DFDC (subset) | Visual — diverse deepfakes | ~100k clips |
| ASVspoof2019 LA | Audio anti-spoofing | 121k utterances |

---

## Setup

### 1. Environment
```bash
conda create -n deepfake python=3.10
conda activate deepfake
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm scikit-learn soundfile opencv-python
pip install -r requirements-api.txt
```

### 2. Train visual models
```bash
python src/train_xception.py --epochs 10
python src/train.py --epochs 10          # EfficientNet-B0
python src/train_vit.py --epochs 10
```

### 3. Train audio model
```bash
python src/train_audio.py --epochs 20 --batch_size 64
```

### 4. Evaluate
```bash
python src/evaluate_ensemble.py
python src/evaluate_audio.py
```

### 5. Run API + UI
```bash
pip install -r requirements-api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict/image` | Visual ensemble inference on image |
| POST | `/predict/audio` | LCNN spoof detection on audio |
| POST | `/predict/video` | Multimodal fusion on video |
| POST | `/predict/gradcam` | Grad-CAM heatmap for image |

Swagger docs: `http://localhost:8000/docs`

---

## Results

### Visual Ensemble (test set — 35,428 samples)
| Metric | Score |
|--------|-------|
| Accuracy | — |
| AUC-ROC | **0.9903** |
| F1-Score | — |

### Audio LCNN (ASVspoof2019 LA)
| Split | Accuracy | AUC-ROC | EER |
|-------|----------|---------|-----|
| Dev | 99.88% | 1.0000 | 0.20% |
| Eval | 86.40% | 0.9808 | 33.26% |

---

## Team

| Name | Role |
|------|------|
| Muavia Shakeel | Model development, backend, UI |
| [Team Member 2] | [Role] |
| [Team Member 3] | [Role] |

**Supervisor:** [Supervisor Name]  
**Institution:** Department of Computer Science, Hazara University Mansehra  
**Degree:** BS Computer Science — Final Year Project 2026

---

## License

For academic use only. Not licensed for commercial deployment.
