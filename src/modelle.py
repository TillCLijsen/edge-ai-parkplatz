"""Erzeugt die verwendeten Modelle mit zwei Ausgabeklassen (frei, belegt)."""

from torch import nn
from torchvision import models


def erstelle_modell(name, vortrainiert=True):
    """Lädt ein ImageNet-vortrainiertes Modell und ersetzt die letzte Schicht."""
    if name == "mobilenet_v3_small":
        gewichte = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if vortrainiert else None
        modell = models.mobilenet_v3_small(weights=gewichte)
        modell.classifier[-1] = nn.Linear(modell.classifier[-1].in_features, 2)
    elif name == "resnet18":
        gewichte = models.ResNet18_Weights.IMAGENET1K_V1 if vortrainiert else None
        modell = models.resnet18(weights=gewichte)
        modell.fc = nn.Linear(modell.fc.in_features, 2)
    else:
        raise ValueError(f"Unbekanntes Modell: {name}")
    return modell
