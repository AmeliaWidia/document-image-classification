"""Arsitektur EfficientNet-B0 dan pemuatan checkpoint eksperimen."""
from pathlib import Path
import torch
from torch import nn
from torchvision import models
from scripts.config import NUM_CLASSES, WEIGHTS_FILENAME, WEIGHTS_PATH


def locate_pretrained_weights(custom_path: str | Path | None = None) -> Path:
    """Temukan file pretrained EfficientNet-B0 di Kaggle Input atau cache."""
    configured = custom_path or WEIGHTS_PATH
    if configured and Path(configured).is_file():
        return Path(configured)
    roots = [Path("/kaggle/input"), Path(torch.hub.get_dir()) / "checkpoints"]
    for root in roots:
        if root.exists():
            matches = list(root.rglob(WEIGHTS_FILENAME))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Bobot pretrained tidak ditemukan: {WEIGHTS_FILENAME}")


def build_model(dropout: float = 0.30, pretrained: bool = True, weights_path=None) -> nn.Module:
    """Buat EfficientNet-B0 dengan classifier 16 kelas dan bobot opsional."""
    model = models.efficientnet_b0(weights=None)
    if pretrained:
        weight_file = locate_pretrained_weights(weights_path)
        state = torch.load(weight_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, NUM_CLASSES))
    return model


def load_checkpoint_model(checkpoint_path: Path, dropout: float, device: torch.device) -> nn.Module:
    """Muat state_dict checkpoint ke model EfficientNet-B0."""
    model = build_model(dropout=dropout, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
