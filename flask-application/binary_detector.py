"""
Binary Mangrove Detector Module

Detects whether an image contains mangrove vegetation or not.
Uses a pre-trained EfficientNet-B0 binary classifier.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

BASE_DIR = Path(__file__).parent
BINARY_MODEL_PATH = BASE_DIR / 'bestBinary.pth'

GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BINARY_LABELS = ['Non-Mangrove', 'Mangrove']

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def build_binary_model(freeze_backbone: bool = True,
                       pretrained_backbone: bool = False) -> nn.Module:
    """Build the binary mangrove detection model using EfficientNet-B0.

    pretrained_backbone=False builds the architecture with randomly
    initialised weights and no network access. This is what you want for
    inference: the saved checkpoint in bestBinary.pth contains every
    backbone parameter, so downloading ImageNet weights first would only
    be overwritten by load_state_dict().

    Set pretrained_backbone=True only when training a new model from
    scratch, where the ImageNet initialisation actually matters.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    model = models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    return model


def load_binary_model():
    """Load the pre-trained binary mangrove detection model."""
    if not BINARY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Binary model file not found: {BINARY_MODEL_PATH}. "
            "Please train and save the binary model first."
        )

    model = build_binary_model(freeze_backbone=True, pretrained_backbone=False).to(GPU)
    model.load_state_dict(
        torch.load(str(BINARY_MODEL_PATH), map_location=GPU, weights_only=True)
    )
    model.eval()
    return model


def extract_tiles(img: Image.Image, tile_size: int = 512, overlap: int = 64):
    """
    Slice a large image into overlapping tiles for prediction.
    Returns a tensor of tiles and their positions.
    """
    W, H = img.size
    step = tile_size - overlap
    tiles, positions = [], []

    for top in range(0, max(H - tile_size + 1, 1), step):
        for left in range(0, max(W - tile_size + 1, 1), step):
            box = (left, top,
                   min(left + tile_size, W),
                   min(top + tile_size, H))
            tile = img.crop(box)
            if tile.size != (tile_size, tile_size):
                padded = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded
            tiles.append(INFERENCE_TRANSFORM(tile))
            positions.append(box)

    return torch.stack(tiles), positions


def predict_binary(image_path: str, model, tile_size: int = 512,
                   overlap: int = 64, batch_size: int = 64,
                   confidence_threshold: float = 0.0) -> dict:
    """
    Run binary mangrove detection on an image.

    Returns a dict with:
        - prediction: 'Mangrove' or 'Non-Mangrove'
        - confidence: confidence score (0-1)
        - probs: [P(Non-Mangrove), P(Mangrove)]
        - n_tiles: number of tiles used
    """
    img = Image.open(image_path).convert('RGB')
    tiles, _ = extract_tiles(img, tile_size, overlap)
    probability = []

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size].to(GPU)
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
            if confidence_threshold > 0:
                probs = probs[probs.max(axis=1) >= confidence_threshold]
            if len(probs):
                probability.append(probs)

    if not probability:
        return {
            'prediction': 'Uncertain',
            'confidence': 0.0,
            'probs': [0.5, 0.5],
            'n_tiles': len(tiles)
        }

    average_prob = np.concatenate(probability, axis=0).mean(axis=0)
    prediction_index = int(average_prob.argmax())
    return {
        'prediction': BINARY_LABELS[prediction_index],
        'confidence': float(average_prob[prediction_index]),
        'probs': average_prob.tolist(),
        'n_tiles': len(tiles)
    }
