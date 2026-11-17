"""Zentrale Einstellungen für das Projekt (Pfade, Seed)."""

import os
import random
from pathlib import Path

import numpy as np

# Projektordner = Ordner über src/
PROJEKT_ORDNER = Path(__file__).resolve().parent.parent
DATEN_ORDNER = PROJEKT_ORDNER / "daten"
ERGEBNIS_ORDNER = PROJEKT_ORDNER / "ergebnisse"
MODELL_ORDNER = PROJEKT_ORDNER / "modelle"
ONNX_ORDNER = MODELL_ORDNER / "onnx"

# Fester Seed, damit Split und Training reproduzierbar sind
SEED = 42


def setze_seeds(seed: int = SEED) -> None:
    """Setzt alle Zufallsgeneratoren auf denselben Startwert."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass  # torch ist für reine Datenskripte nicht nötig
