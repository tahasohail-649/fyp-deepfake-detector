import os
import torch
import torch.nn as nn
import timm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

MODEL_CONFIGS = {
    "xception":        ("xception",               "xception_best.pth"),
    "efficientnet_b0": ("efficientnet_b0",         "efficientnet_b0_best.pth"),
    "vit":             ("vit_small_patch16_224",   "vit_best.pth"),
}

# AUC-based weights (tune after evaluating individual models)
DEFAULT_WEIGHTS = {
    "xception":        1.0,
    "efficientnet_b0": 1.0,
    "vit":             1.0,
}


def load_model(timm_name: str, ckpt_path: str, device: torch.device) -> nn.Module:
    model = timm.create_model(timm_name, pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_ensemble(
    weights: dict = None,
    device: torch.device = None,
) -> tuple[dict, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weights is None:
        weights = DEFAULT_WEIGHTS

    models = {}
    for name, (timm_name, ckpt_file) in MODEL_CONFIGS.items():
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        models[name] = load_model(timm_name, ckpt_path, device)

    return models, weights


@torch.no_grad()
def predict_proba(
    imgs: torch.Tensor,
    models: dict,
    weights: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns weighted-average fake probability for a batch of images.
    imgs: (N, 3, 224, 224) tensor, already normalized
    Returns: (N,) tensor of fake probabilities in [0, 1]
    """
    imgs = imgs.to(device)
    total_weight = sum(weights.values())
    ensemble_prob = torch.zeros(imgs.size(0), device=device)

    for name, model in models.items():
        logits = model(imgs).squeeze(1)          # (N,)
        prob = torch.sigmoid(logits)             # (N,)
        ensemble_prob += (weights[name] / total_weight) * prob

    return ensemble_prob.cpu()


@torch.no_grad()
def predict_single(
    img_tensor: torch.Tensor,
    models: dict,
    weights: dict,
    device: torch.device,
) -> dict:
    """
    Single image inference. img_tensor: (3, 224, 224) or (1, 3, 224, 224).
    Returns dict with per-model probs + ensemble prob + label.
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    img_tensor = img_tensor.to(device)
    total_weight = sum(weights.values())
    result = {}
    ensemble_prob = 0.0

    for name, model in models.items():
        logit = model(img_tensor).squeeze()
        prob = torch.sigmoid(logit).item()
        result[f"{name}_prob"] = round(prob, 4)
        ensemble_prob += (weights[name] / total_weight) * prob

    result["ensemble_prob"] = round(ensemble_prob, 4)
    result["label"] = "FAKE" if ensemble_prob >= 0.5 else "REAL"
    result["confidence"] = round(
        ensemble_prob if ensemble_prob >= 0.5 else 1.0 - ensemble_prob, 4
    )
    return result
