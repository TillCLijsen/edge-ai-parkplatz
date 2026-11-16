"""PyTorch-Datensatz für die vorberechneten Ausschnitte."""

import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ausschnitte_vorberechnen import CACHE_ORDNER
from src.konfiguration import SEED
from src.vorverarbeitung import normalisiere


def augmentiere(bild, zufall):
    """Zufällige Veränderung eines uint8-Ausschnitts (64, 64, 3), nur im Training.

    - horizontal spiegeln: Autos stehen mal links-, mal rechtsbündig
    - um 180° drehen: die Fahrtrichtung im Ausschnitt ist nicht festgelegt
    - Helligkeit und Kontrast: Sonne, Wolken und Regen verändern die Belichtung
    """
    if zufall.random() < 0.5:
        bild = bild[:, ::-1]
    if zufall.random() < 0.5:
        bild = bild[::-1, ::-1]
    werte = bild.astype(np.float32)
    helligkeit = zufall.uniform(0.7, 1.3)
    kontrast = zufall.uniform(0.7, 1.3)
    mittel = werte.mean()
    werte = (werte - mittel) * kontrast + mittel * helligkeit
    return np.clip(werte, 0, 255).astype(np.uint8)


class PlatzDatensatz(Dataset):
    """Liefert (Tensor 3×64×64, Label) für eine Menge aus daten/cache/."""

    def __init__(self, menge, augmentieren=False, seed=SEED):
        # mmap: die Datei wird nicht komplett in den Arbeitsspeicher geladen
        self.bilder = np.load(CACHE_ORDNER / f"{menge}_bilder.npy", mmap_mode="r")
        with open(CACHE_ORDNER / f"{menge}_info.csv", encoding="utf-8") as datei:
            self.labels = np.array([int(z["label"]) for z in csv.DictReader(datei)], dtype=np.int64)
        assert len(self.bilder) == len(self.labels)
        self.augmentieren = augmentieren
        self.zufall = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        bild = np.array(self.bilder[index])
        if self.augmentieren:
            bild = augmentiere(bild, self.zufall)
        return torch.from_numpy(normalisiere(bild)), int(self.labels[index])
