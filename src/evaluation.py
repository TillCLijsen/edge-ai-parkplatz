"""Bewertet trainierte PyTorch-Modelle auf Test A (UFPR, ungesehene Tage)
und Test B (PUCPR, fremder Parkplatz).

Wichtig: Dieses Skript wird erst ausgeführt, wenn alle Trainingseinstellungen
feststehen. Die Testmengen dienen nur der Berichterstattung, nie der Auswahl (E12).

Kennzahlen pro Menge, Parkplatz und Wetter: Accuracy, Balanced Accuracy,
F1 (belegt), Konfusionsmatrix. Für die Balanced Accuracy wird ein 95-%-
Konfidenzintervall per Bootstrap über Aufnahmetage berechnet, weil Ausschnitte
desselben Tages nicht unabhängig voneinander sind.

Aufruf:  python -m src.evaluation                      (alle Modelle in modelle/)
         python -m src.evaluation --modelle mobilenet_v3_small_seed42
"""

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datensatz import PlatzDatensatz
from src.konfiguration import DATEN_ORDNER, ERGEBNIS_ORDNER, MODELL_ORDNER, SEED
from src.metriken import werte_aus
from src.modelle import erstelle_modell
from src.training import waehle_geraet

TESTMENGEN = ["test_a", "test_b"]


@torch.no_grad()
def sage_vorher(modell, menge, geraet):
    """Wahrscheinlichkeit für 'belegt' für jeden Ausschnitt einer Menge."""
    lader = DataLoader(PlatzDatensatz(menge), batch_size=1024)
    modell.eval()
    wahrscheinlichkeiten = []
    for x, _ in lader:
        ausgabe = torch.softmax(modell(x.to(geraet)), dim=1)[:, 1]
        wahrscheinlichkeiten.append(ausgabe.cpu().numpy())
    return np.concatenate(wahrscheinlichkeiten)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelle", nargs="+",
                        default=sorted(p.stem for p in MODELL_ORDNER.glob("*.pt")))
    args = parser.parse_args()

    geraet = waehle_geraet()
    zufall = np.random.default_rng(SEED)
    vorhersage_ordner = DATEN_ORDNER / "vorhersagen"
    vorhersage_ordner.mkdir(parents=True, exist_ok=True)
    alle_zeilen = []

    for name in args.modelle:
        architektur = name.rsplit("_seed", 1)[0]
        modell = erstelle_modell(architektur, vortrainiert=False)
        modell.load_state_dict(torch.load(MODELL_ORDNER / f"{name}.pt", map_location="cpu"))
        modell.to(geraet)
        for menge in TESTMENGEN:
            wahrscheinlichkeiten = sage_vorher(modell, menge, geraet)
            np.save(vorhersage_ordner / f"{name}_{menge}.npy", wahrscheinlichkeiten.astype(np.float32))
            zeilen = werte_aus(name, menge, wahrscheinlichkeiten, zufall)
            alle_zeilen.extend(zeilen)
            for z in zeilen:
                if z["gruppe"] in ("gesamt",):
                    print(f"{name:28} {menge}: Acc {z['accuracy']:.4f}  BalAcc {z['balanced_accuracy']:.4f} "
                          f"[{z['bal_acc_ki_unten']:.4f}; {z['bal_acc_ki_oben']:.4f}]  F1 {z['f1_belegt']:.4f}")

    ziel = ERGEBNIS_ORDNER / "evaluation_pytorch.csv"
    with open(ziel, "w", newline="", encoding="utf-8") as datei:
        schreiber = csv.DictWriter(datei, fieldnames=list(alle_zeilen[0].keys()))
        schreiber.writeheader()
        schreiber.writerows(alle_zeilen)
    print("Gespeichert:", ziel)


if __name__ == "__main__":
    main()
