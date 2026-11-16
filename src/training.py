"""Trainiert ein Modell auf den Trainingsdaten und wählt die beste Epoche
anhand der Balanced Accuracy auf der Validierungsmenge.

Aufruf:  python -m src.training --modell mobilenet_v3_small --seed 42

Ausgaben:
- modelle/<modell>_seed<seed>.pt          bestes Modell (state_dict)
- ergebnisse/training/<modell>_seed<seed>.csv   Verlauf pro Epoche
"""

import argparse
import csv
import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datensatz import PlatzDatensatz
from src.konfiguration import ERGEBNIS_ORDNER, MODELL_ORDNER, setze_seeds
from src.modelle import erstelle_modell


def waehle_geraet():
    """Training auf der Apple-GPU (MPS), falls vorhanden, sonst CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def bewerte(modell, lader, geraet):
    """Gibt Accuracy und Balanced Accuracy auf einer Menge zurück."""
    modell.eval()
    wahr, vorhersage = [], []
    for x, y in lader:
        ausgabe = modell(x.to(geraet))
        vorhersage.append(ausgabe.argmax(dim=1).cpu().numpy())
        wahr.append(y.numpy())
    wahr, vorhersage = np.concatenate(wahr), np.concatenate(vorhersage)
    return accuracy_score(wahr, vorhersage), balanced_accuracy_score(wahr, vorhersage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modell", default="mobilenet_v3_small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochen", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lernrate", type=float, default=1e-3)
    parser.add_argument("--geduld", type=int, default=3,
                        help="Abbruch, wenn sich die Validierung so viele Epochen nicht verbessert")
    args = parser.parse_args()

    setze_seeds(args.seed)
    geraet = waehle_geraet()
    name = f"{args.modell}_seed{args.seed}"
    print(f"Training {name} auf {geraet}")

    training = PlatzDatensatz("training", augmentieren=True, seed=args.seed)
    validierung = PlatzDatensatz("validierung")
    lader_training = DataLoader(training, batch_size=args.batch, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed))
    lader_validierung = DataLoader(validierung, batch_size=512)
    print(f"Training: {len(training)} Ausschnitte, Validierung: {len(validierung)}")

    modell = erstelle_modell(args.modell).to(geraet)
    optimierer = torch.optim.AdamW(modell.parameters(), lr=args.lernrate, weight_decay=1e-4)
    planer = torch.optim.lr_scheduler.CosineAnnealingLR(optimierer, T_max=args.epochen)
    verlustfunktion = nn.CrossEntropyLoss()

    MODELL_ORDNER.mkdir(exist_ok=True)
    log_ordner = ERGEBNIS_ORDNER / "training"
    log_ordner.mkdir(parents=True, exist_ok=True)
    verlauf = []
    beste, ohne_verbesserung = -1.0, 0
    start_gesamt = time.time()

    for epoche in range(1, args.epochen + 1):
        start = time.time()
        modell.train()
        summe_verlust = 0.0
        for x, y in tqdm(lader_training, desc=f"Epoche {epoche}", leave=False):
            x, y = x.to(geraet), y.to(geraet)
            optimierer.zero_grad()
            verlust = verlustfunktion(modell(x), y)
            verlust.backward()
            optimierer.step()
            summe_verlust += verlust.item() * len(y)
        planer.step()

        acc, bal_acc = bewerte(modell, lader_validierung, geraet)
        eintrag = {"epoche": epoche, "verlust_training": round(summe_verlust / len(training), 5),
                   "val_accuracy": round(acc, 5), "val_balanced_accuracy": round(bal_acc, 5),
                   "sekunden": round(time.time() - start, 1)}
        verlauf.append(eintrag)
        print(eintrag)

        if bal_acc > beste:
            beste, ohne_verbesserung = bal_acc, 0
            torch.save(modell.state_dict(), MODELL_ORDNER / f"{name}.pt")
        else:
            ohne_verbesserung += 1
            if ohne_verbesserung >= args.geduld:
                print(f"Kein Fortschritt seit {args.geduld} Epochen, Abbruch.")
                break

    with open(log_ordner / f"{name}.csv", "w", newline="", encoding="utf-8") as datei:
        schreiber = csv.DictWriter(datei, fieldnames=list(verlauf[0].keys()))
        schreiber.writeheader()
        schreiber.writerows(verlauf)
    zusammenfassung = {"modell": args.modell, "seed": args.seed, "geraet": str(geraet),
                       "beste_val_balanced_accuracy": round(beste, 5),
                       "epochen_gelaufen": len(verlauf),
                       "minuten_gesamt": round((time.time() - start_gesamt) / 60, 1),
                       "parameter": vars(args), "torch": torch.__version__}
    with open(log_ordner / f"{name}.json", "w", encoding="utf-8") as datei:
        json.dump(zusammenfassung, datei, indent=2, ensure_ascii=False)
    print(zusammenfassung)


if __name__ == "__main__":
    main()
