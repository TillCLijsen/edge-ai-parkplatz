"""Kennzahlen für die Klassifikation, ohne Abhängigkeit von PyTorch.

Wird von src/evaluation.py (PyTorch) und src/messung.py (ONNX) gemeinsam genutzt,
damit beide exakt gleich rechnen.
"""

import csv
from collections import defaultdict

import numpy as np

from src.ausschnitte_vorberechnen import CACHE_ORDNER

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
                       "bal_acc_ki_unten": round(float(unten), 5), "bal_acc_ki_oben": round(float(oben), 5),
                       "f1_belegt": round(f1, 5),
                       "fehler_pro_10000": round((fp + fn) / (tp + tn + fp + fn) * 10000, 2),
                       "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return zeilen
