"""
Multimodal fusion: combine visual ensemble + audio LCNN scores.

Two fusion modes:
  - weighted_avg: w_visual * visual_prob + w_audio * audio_prob
  - max_suspicion: max(visual_prob, audio_prob)  — flag fake if either modality suspicious

Usage for single video file:
    from src.multimodal_fusion import MultimodalDetector
    detector = MultimodalDetector()
    result = detector.predict_video(video_path)

Usage for image (no audio):
    result = detector.predict_image(img_tensor)
"""

import os
import sys
import tempfile
import subprocess

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ensemble import load_ensemble, predict_single
from src.audio_classifier import load_audio_model, predict_audio_file
from src.audio_dataset import wav_to_logmel, load_waveform

VISUAL_WEIGHT = 0.6   # visual ensemble carries more weight (3 models vs 1)
AUDIO_WEIGHT  = 0.4

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class MultimodalDetector:
    def __init__(
        self,
        visual_weights: dict = None,
        visual_weight: float = VISUAL_WEIGHT,
        audio_weight: float  = AUDIO_WEIGHT,
        fusion_mode: str = "weighted_avg",   # "weighted_avg" | "max_suspicion"
        device: torch.device = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.fusion_mode   = fusion_mode
        self.visual_weight = visual_weight
        self.audio_weight  = audio_weight

        self.visual_models, self.visual_weights = load_ensemble(
            weights=visual_weights, device=device
        )
        try:
            self.audio_model = load_audio_model(device=device)
            self.has_audio = True
        except FileNotFoundError:
            self.audio_model = None
            self.has_audio = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, img_tensor: torch.Tensor) -> dict:
        """
        Visual-only prediction.
        img_tensor: (3, 224, 224) or (1, 3, 224, 224), normalized.
        """
        result = predict_single(
            img_tensor, self.visual_models, self.visual_weights, self.device
        )
        result["audio_available"] = False
        result["fusion_mode"] = "visual_only"
        return result

    def predict_image_path(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        tensor = VAL_TRANSFORMS(img)
        return self.predict_image(tensor)

    def predict_video(self, video_path: str, sample_frames: int = 10) -> dict:
        """
        Full multimodal prediction on a video file.
        Extracts frames for visual, extracts audio for LCNN.
        Requires ffmpeg on PATH.
        """
        visual_prob = self._visual_from_video(video_path, sample_frames)

        audio_prob = None
        if self.has_audio:
            audio_prob = self._audio_from_video(video_path)

        return self._fuse(visual_prob, audio_prob, video_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _visual_from_video(self, video_path: str, n_frames: int) -> float:
        frames = _extract_frames_ffmpeg(video_path, n_frames)
        if not frames:
            return 0.5  # uncertain if no frames extracted

        probs = []
        for frame_path in frames:
            try:
                img = Image.open(frame_path).convert("RGB")
                tensor = VAL_TRANSFORMS(img)
                res = predict_single(
                    tensor, self.visual_models, self.visual_weights, self.device
                )
                probs.append(res["ensemble_prob"])
            except Exception:
                continue
            finally:
                try:
                    os.remove(frame_path)
                except OSError:
                    pass

        return sum(probs) / len(probs) if probs else 0.5

    def _audio_from_video(self, video_path: str) -> float | None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                 wav_path],
                capture_output=True, timeout=60,
            )
            if ret.returncode != 0 or not os.path.exists(wav_path):
                return None
            result = predict_audio_file(wav_path, self.audio_model, self.device)
            return result["spoof_prob"]
        except Exception:
            return None
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def _fuse(
        self,
        visual_prob: float,
        audio_prob: float | None,
        source_path: str,
    ) -> dict:
        if audio_prob is None or not self.has_audio:
            ensemble_prob = visual_prob
            fusion_used   = "visual_only"
        elif self.fusion_mode == "max_suspicion":
            ensemble_prob = max(visual_prob, audio_prob)
            fusion_used   = "max_suspicion"
        else:
            total_w = self.visual_weight + self.audio_weight
            ensemble_prob = (
                self.visual_weight * visual_prob + self.audio_weight * audio_prob
            ) / total_w
            fusion_used = "weighted_avg"

        label      = "FAKE" if ensemble_prob >= 0.5 else "REAL"
        confidence = ensemble_prob if ensemble_prob >= 0.5 else 1.0 - ensemble_prob

        return {
            "source":        source_path,
            "label":         label,
            "confidence":    round(confidence,    4),
            "ensemble_prob": round(ensemble_prob, 4),
            "visual_prob":   round(visual_prob,   4),
            "audio_prob":    round(audio_prob, 4) if audio_prob is not None else None,
            "audio_available": audio_prob is not None,
            "fusion_mode":   fusion_used,
        }


# ------------------------------------------------------------------
# ffmpeg frame extraction
# ------------------------------------------------------------------

def _extract_frames_ffmpeg(video_path: str, n_frames: int) -> list[str]:
    """Extract n_frames evenly-spaced frames from video. Returns list of temp PNG paths."""
    tmpdir = tempfile.mkdtemp()
    out_pattern = os.path.join(tmpdir, "frame_%04d.png")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", f"select=not(mod(n\\,{max(1, n_frames)}))",
             "-vsync", "vfr", "-frames:v", str(n_frames),
             out_pattern],
            capture_output=True, timeout=120,
        )
    except Exception:
        return []
    return sorted(
        [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".png")]
    )[:n_frames]
