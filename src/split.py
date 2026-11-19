"""Teilt die Aufnahmetage von PKLot in Trainings-, Validierungs- und Testmengen.

Regeln (stehen im docs/ENTSCHEIDUNGEN.md, E5 und E8):
- Bilder von einem Tag landen immer komplett in derselben Menge.
- UFPR04 und UFPR05: 50 % der Tage Test A, vom Rest ca. 20 % Validierung.
- PUCPR: 50 % der Tage Test B, der Rest wird nicht verwendet
  (Anpassungsversuch gestrichen, E13).
- Regentage werden zuerst verteilt, damit jede Testmenge Regen enthält um verälschte Resultate zu verhindern.

Eingabe:  ergebnisse/datenuebersicht_bilder.csv (aus src/datenuebersicht.py)
Ausgabe:  ergebnisse/split_tage.csv (Parkplatz, Tag, Regentag, Menge)
"""

import csv
import math
import random
from collections import defaultdict

from src.konfiguration import ERGEBNIS_ORDNER, SEED

ANTEIL_TEST = 0.5
ANTEIL_VALIDIERUNG = 0.2  # bezogen auf die Tage, die nicht im Test sind


def lade_bilder():
    """Liest die Bildübersicht ein."""
    with open(ERGEBNIS_ORDNER / "datenuebersicht_bilder.csv", encoding="utf-8") as datei:
        return list(csv.DictReader(datei))


def teile_tage(tage, anteil, zufall):
    """Teilt eine Liste von (Tag, Regentag) geschichtet nach Regen auf.

    Gibt (ausgewählte Tage, restliche Tage) zurück. Von den Regentagen wird
    aufgerundet, damit die ausgewählte Menge sicher Regen enthält.
    """
    regen = sorted(t for t, r in tage if r)
    trocken = sorted(t for t, r in tage if not r)
    zufall.shuffle(regen)
    zufall.shuffle(trocken)
    n_regen = math.ceil(len(regen) * anteil) if len(regen) >= 2 else 0
    n_trocken = round(len(tage) * anteil) - n_regen
    n_trocken = max(1, min(n_trocken, len(trocken)))
    auswahl = regen[:n_regen] + trocken[:n_trocken]
    rest = regen[n_regen:] + trocken[n_trocken:]
    return auswahl, rest


def main():
    zufall = random.Random(SEED)
    bilder = lade_bilder()

    # Pro Parkplatz: Tag -> Regentag ja/nein (Regentag = mindestens ein Regenbild)
    regentag = defaultdict(dict)
    for b in bilder:
        tage = regentag[b["parkplatz"]]
        tage[b["tag"]] = tage.get(b["tag"], False) or b["wetter"] == "Rainy"

    zuordnung = {}  # (Parkplatz, Tag) -> Menge
    for parkplatz in sorted(regentag):
        tage = sorted(regentag[parkplatz].items())
        test, rest = teile_tage(tage, ANTEIL_TEST, zufall)
        if parkplatz == "PUCPR":
            mengen = {"test_b": test, "anpassung": rest}
        else:
            rest_mit_regen = [(t, regentag[parkplatz][t]) for t in rest]
            validierung, training = teile_tage(rest_mit_regen, ANTEIL_VALIDIERUNG, zufall)
            mengen = {"test_a": test, "validierung": validierung, "training": training}
        for menge, liste in mengen.items():
            for tag in liste:
                zuordnung[(parkplatz, tag)] = menge

    ziel = ERGEBNIS_ORDNER / "split_tage.csv"
    with open(ziel, "w", newline="", encoding="utf-8") as datei:
        schreiber = csv.writer(datei)
        schreiber.writerow(["parkplatz", "tag", "regentag", "menge"])
        for (parkplatz, tag), menge in sorted(zuordnung.items()):
            schreiber.writerow([parkplatz, tag, int(regentag[parkplatz][tag]), menge])

    # Zusammenfassung zur Kontrolle
    statistik = defaultdict(lambda: defaultdict(int))
    for b in bilder:
        menge = zuordnung[(b["parkplatz"], b["tag"])]
        s = statistik[(b["parkplatz"], menge)]
        s["bilder"] += 1
        s["frei"] += int(b["frei"])
        s["belegt"] += int(b["belegt"])
        s["bilder_" + b["wetter"]] += 1
    print(f"{'Parkplatz':8} {'Menge':12} {'Tage':>4} {'Regen':>5} {'Bilder':>6} "
          f"{'Plätze':>7} {'belegt':>7}  Bilder Sonne/Wolken/Regen")
    for (parkplatz, menge), s in sorted(statistik.items()):
        tage = [t for (p, t), m in zuordnung.items() if p == parkplatz and m == menge]
        n_regen = sum(regentag[parkplatz][t] for t in tage)
        plaetze = s["frei"] + s["belegt"]
        print(f"{parkplatz:8} {menge:12} {len(tage):>4} {n_regen:>5} {s['bilder']:>6} "
              f"{plaetze:>7} {s['belegt'] / plaetze:>7.1%}  "
              f"{s['bilder_Sunny']}/{s['bilder_Cloudy']}/{s['bilder_Rainy']}")
    print("Gespeichert:", ziel)


if __name__ == "__main__":
    main()
