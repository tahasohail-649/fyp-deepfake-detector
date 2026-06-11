import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio_dataset import load_waveform, wav_to_logmel
from src.train_audio import LCNN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
AUDIO_CKPT = os.path.join(CHECKPOINT_DIR, "audio_lcnn_best.pth")


def load_audio_model(device: torch.device = None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(AUDIO_CKPT):
        raise FileNotFoundError(f"Audio checkpoint not found: {AUDIO_CKPT}")
    model = LCNN()
    ckpt = torch.load(AUDIO_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_audio_file(
    audio_path: str,
    model: nn.Module,
    device: torch.device,
) -> dict:
    """
    Run inference on a single audio file (.flac, .wav, .mp3, etc.).
    Returns dict with spoof probability, label, and confidence.
    """
    wav = load_waveform(audio_path)
    logmel = wav_to_logmel(wav).unsqueeze(0).to(device)  # (1, 1, 80, 300)

    logit = model(logmel).squeeze()
    prob  = torch.sigmoid(logit).item()

    return {
        "spoof_prob":  round(prob, 4),
        "label":       "SPOOF" if prob >= 0.5 else "BONAFIDE",
        "confidence":  round(prob if prob >= 0.5 else 1.0 - prob, 4),
    }


@torch.no_grad()
def predict_audio_batch(
    specs: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Batch inference on pre-computed log-mel spectrograms.
    specs: (N, 1, 80, 300)
    Returns: (N,) spoof probability tensor on CPU.
    """
    specs = specs.to(device)
    logits = model(specs).squeeze(1)
    return torch.sigmoid(logits).cpu()
