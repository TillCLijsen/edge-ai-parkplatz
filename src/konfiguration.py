"""Zentrale Einstellungen für das Projekt (Pfade, Seed)."""

import os
import random
import sys
from datetime import datetime
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




class _Doppelt:
    """Schreibt Text gleichzeitig in die Konsole und in eine Datei."""

    def __init__(self, konsole, datei):
        self.konsole = konsole
        self.datei = datei

    def write(self, text):
        self.konsole.write(text)
        self.datei.write(text)
        self.datei.flush()

    def flush(self):
        self.konsole.flush()
        self.datei.flush()


def starte_protokoll(name: str) -> None:
    """Speichert alle print-Ausgaben zusätzlich in ergebnisse/protokolle/<name>.log."""
    ordner = ERGEBNIS_ORDNER / "protokolle"
    ordner.mkdir(parents=True, exist_ok=True)
    datei = open(ordner / f"{name}.log", "a", encoding="utf-8")
    datei.write(f"\n=== Start {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    sys.stdout = _Doppelt(sys.__stdout__, datei)