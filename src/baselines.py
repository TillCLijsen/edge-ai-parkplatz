"""Mehrheitsklassen-Baseline, berechnet auf den tatsächlichen Testdaten.

Zwei Varianten werden berichtet:
1. Mehrheitsklasse aus dem Training, angewendet auf die Testdaten
   (das, was ein trivialer Klassifikator ohne Blick auf den Test leisten würde).
2. Anteil der häufigsten Klasse in der jeweiligen Testmenge
   (Obergrenze für jede konstante Vorhersage).

Eingabe:  daten/cache/<menge>_info.csv (aus src/ausschnitte_vorberechnen.py)
Ausgabe:  ergebnisse/baselines.csv
"""

import csv
from collections import defaultdict

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.ausschnitte_vorberechnen import CACHE_ORDNER
from src.konfiguration import ERGEBNIS_ORDNER

TESTMENGEN = ["test_a", "test_b"]
KLASSEN = {0: "frei", 1: "belegt"}


def lese_labels(menge):
    with open(CACHE_ORDNER / f"{menge}_info.csv", encoding="utf-8") as datei:
        return list(csv.DictReader(datei))


def main():
    training = [int(z["label"]) for z in lese_labels("training")]
    mehrheit_training = int(sum(training) > len(training) / 2)
    print(f"Mehrheitsklasse im Training: {KLASSEN[mehrheit_training]} "
          f"({max(sum(training), len(training) - sum(training)) / len(training):.1%})")

    zeilen = []
    for menge in TESTMENGEN:
        info = lese_labels(menge)
        gruppen = defaultdict(list)
        for z in info:
            label = int(z["label"])
            gruppen["gesamt"].append(label)
            gruppen[z["parkplatz"]].append(label)
            gruppen[z["wetter"]].append(label)
        for gruppe, labels in gruppen.items():
            vorhersage = [mehrheit_training] * len(labels)
            anteil_belegt = sum(labels) / len(labels)
            zeilen.append({
                "menge": menge,
                "gruppe": gruppe,
                "anzahl": len(labels),
                "anteil_belegt": round(anteil_belegt, 4),
                "accuracy_mehrheit_training": round(accuracy_score(labels, vorhersage), 4),
                "balanced_accuracy_mehrheit_training": round(balanced_accuracy_score(labels, vorhersage), 4),
                "f1_belegt_mehrheit_training": round(f1_score(labels, vorhersage, zero_division=0), 4),
                "accuracy_beste_konstante": round(max(anteil_belegt, 1 - anteil_belegt), 4),
            })

    ziel = ERGEBNIS_ORDNER / "baselines.csv"
    with open(ziel, "w", newline="", encoding="utf-8") as datei:
        schreiber = csv.DictWriter(datei, fieldnames=list(zeilen[0].keys()))
        schreiber.writeheader()
        schreiber.writerows(zeilen)
    for z in zeilen:
        print(f"{z['menge']:7} {z['gruppe']:8} n={z['anzahl']:>7}  belegt {z['anteil_belegt']:.1%}  "
              f"Acc {z['accuracy_mehrheit_training']:.3f}  "
              f"BalAcc {z['balanced_accuracy_mehrheit_training']:.3f}  "
              f"beste Konstante {z['accuracy_beste_konstante']:.3f}")
    print("Gespeichert:", ziel)


if __name__ == "__main__":
    main()
