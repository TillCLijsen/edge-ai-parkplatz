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
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ausschnitte_vorberechnen import CACHE_ORDNER
from src.datensatz import PlatzDatensatz
from src.konfiguration import DATEN_ORDNER, ERGEBNIS_ORDNER, MODELL_ORDNER, SEED
from src.modelle import erstelle_modell
from src.training import waehle_geraet

TESTMENGEN = ["test_a", "test_b"]
BOOTSTRAP_WIEDERHOLUNGEN = 1000


def kennzahlen(tp, tn, fp, fn):
    """Accuracy, Balanced Accuracy und F1 (Klasse belegt) aus der Konfusionsmatrix."""
    n = tp + tn + fp + fn
    sensitivitaet = tp / (tp + fn) if tp + fn else 0.0   # belegte Plätze richtig erkannt
    spezifitaet = tn / (tn + fp) if tn + fp else 0.0     # freie Plätze richtig erkannt
    praezision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * praezision * sensitivitaet / (praezision + sensitivitaet) if praezision + sensitivitaet else 0.0
    return (tp + tn) / n, (sensitivitaet + spezifitaet) / 2, f1


def bootstrap_tage(zaehler_pro_tag, zufall):
    """95-%-Intervall der Balanced Accuracy, Tage mit Zurücklegen gezogen."""
    tage = np.array(list(zaehler_pro_tag.values()))  # Zeilen: tp, tn, fp, fn
    werte = []
    for _ in range(BOOTSTRAP_WIEDERHOLUNGEN):
        auswahl = tage[zufall.integers(0, len(tage), len(tage))].sum(axis=0)
        werte.append(kennzahlen(*auswahl)[1])
    return np.percentile(werte, [2.5, 97.5])


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


def werte_aus(name, menge, wahrscheinlichkeiten, zufall):
    """Kennzahlen für die ganze Menge sowie pro Parkplatz und Wetter."""
    with open(CACHE_ORDNER / f"{menge}_info.csv", encoding="utf-8") as datei:
        info = list(csv.DictReader(datei))
    vorhersage = (wahrscheinlichkeiten >= 0.5).astype(int)

    # Zähler pro Gruppe und Tag: [tp, tn, fp, fn]
    zaehler = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=np.int64)))
    for z, v in zip(info, vorhersage):
        wahr = int(z["label"])
        feld = 0 if (wahr, v) == (1, 1) else 1 if (wahr, v) == (0, 0) else 2 if v == 1 else 3
        tag = (z["parkplatz"], z["tag"])
        for gruppe in ("gesamt", z["parkplatz"], z["wetter"]):
            zaehler[gruppe][tag][feld] += 1

    zeilen = []
    for gruppe, pro_tag in zaehler.items():
        tp, tn, fp, fn = (int(x) for x in sum(pro_tag.values()))
        acc, bal_acc, f1 = kennzahlen(tp, tn, fp, fn)
        unten, oben = bootstrap_tage(pro_tag, zufall)
        zeilen.append({"modell": name, "menge": menge, "gruppe": gruppe,
                       "anzahl": tp + tn + fp + fn, "tage": len(pro_tag),
                       "accuracy": round(acc, 5), "balanced_accuracy": round(bal_acc, 5),
                       "bal_acc_ki_unten": round(unten, 5), "bal_acc_ki_oben": round(oben, 5),
                       "f1_belegt": round(f1, 5), "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return zeilen


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
