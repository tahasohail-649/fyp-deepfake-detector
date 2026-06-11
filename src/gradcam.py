"""
Grad-CAM for visual ensemble models.

Targets last conv layer of each backbone via forward/backward hooks.
Returns a base64-encoded PNG of the heatmap overlaid on the original image.
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def _get_target_layer(model, model_name: str):
    """Return the last conv/feature layer for gradient hooks."""
    if model_name == "xception":
        # timm xception: model.act4 is last activation before global pool
        return model.act4
    elif model_name == "efficientnet_b0":
        # timm efficientnet: model.conv_head is last conv
        return model.conv_head
    elif model_name.startswith("vit"):
        # ViT: last norm layer before head
        return model.norm
    return list(model.children())[-2]   # fallback


class GradCAM:
    def __init__(self, model: torch.nn.Module, model_name: str, device: torch.device):
        self.model      = model
        self.model_name = model_name
        self.device     = device
        self._acts      = None
        self._grads     = None
        self._hooks     = []

        target = _get_target_layer(model, model_name)
        self._hooks.append(
            target.register_forward_hook(self._save_acts)
        )
        self._hooks.append(
            target.register_full_backward_hook(self._save_grads)
        )

    def _save_acts(self, _module, _inp, output):
        self._acts = output.detach()

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        img_tensor: (1, 3, 224, 224) normalized, on CPU.
        Returns cam: (224, 224) float32 in [0, 1].
        """
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)

        logit = self.model(x)          # (1, num_classes)
        # For binary models (num_classes=1) just use logit directly
        if logit.shape[-1] == 1:
            score = logit[0, 0]
        else:
            score = logit[0, logit.argmax(dim=1).item()]

        self.model.zero_grad()
        score.backward()

        acts  = self._acts  # (1, C, H, W) or (1, N, C) for ViT
        grads = self._grads

        if acts is None or grads is None:
            return np.zeros((224, 224), dtype=np.float32)

        # ViT outputs (1, N, C) — average over tokens
        if acts.dim() == 3:
            weights = grads.mean(dim=1, keepdim=True)   # (1, 1, C)
            cam = (weights * acts).sum(dim=2)            # (1, N)
            # reshape to rough spatial grid
            n_patches = cam.shape[1]
            h = w = int(n_patches ** 0.5)
            cam = cam[:, :h*w].reshape(1, 1, h, w)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def overlay_heatmap(cam: np.ndarray, original_img: Image.Image, alpha: float = 0.45) -> str:
    """
    Blend Grad-CAM heatmap onto original PIL image.
    Returns base64-encoded PNG string.
    """
    img_np = np.array(original_img.resize((224, 224))).astype(np.uint8)

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_gradcam(
    image_path: str,
    models: dict,
    model_name: str,
    device: torch.device,
    val_transforms,
) -> str:
    """
    Full pipeline: load image → Grad-CAM → overlay → base64 PNG.

    model_name: one of "xception", "efficientnet_b0", "vit"
    Returns base64 PNG string.
    """
    from PIL import Image as PILImage

    pil_img = PILImage.open(image_path).convert("RGB")
    tensor  = val_transforms(pil_img)   # (3, 224, 224)

    model = models[model_name]
    gc    = GradCAM(model, model_name, device)
    try:
        cam = gc(tensor)
    finally:
        gc.remove_hooks()

    return overlay_heatmap(cam, pil_img)
