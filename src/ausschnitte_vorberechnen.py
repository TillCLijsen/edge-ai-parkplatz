"""Berechnet die 64×64-Ausschnitte aller gelabelten Stellplätze einmal vor.

Für jede Menge entstehen in daten/cache/:
- <menge>_bilder.npy : uint8-Array (N, 64, 64, 3)
- <menge>_info.csv   : eine Zeile pro Ausschnitt (Parkplatz, Wetter, Tag, Datei, Platz, Label)

Es werden dieselben Funktionen wie im Live-Betrieb verwendet
(schneide_platz_aus und skaliere aus src/vorverarbeitung.py).

Aufruf:  python -m src.ausschnitte_vorberechnen
         python -m src.ausschnitte_vorberechnen --mengen anpassung
"""

import argparse
import csv
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from src.konfiguration import DATEN_ORDNER, ERGEBNIS_ORDNER
from src.vorverarbeitung import EINGABE_GROESSE, lese_plaetze, schneide_platz_aus, skaliere

PKLOT = DATEN_ORDNER / "PKLot" / "PKLot"
CACHE_ORDNER = DATEN_ORDNER / "cache"
STANDARD_MENGEN = ["training", "validierung", "test_a", "test_b"]


def verarbeite_bild(zeile):
    """Schneidet alle gelabelten Plätze eines Bildes aus (läuft in einem Unterprozess)."""
    ordner = PKLOT / zeile["parkplatz"] / zeile["wetter"] / zeile["tag"]
    bild = Image.open(ordner / zeile["datei"]).convert("RGB")
    ausschnitte, infos = [], []
    for platz in lese_plaetze(ordner / zeile["datei"].replace(".jpg", ".xml")):
        if platz["label"] is None:
            continue  # E9: Plätze ohne Label werden ausgeschlossen
        ausschnitte.append(skaliere(schneide_platz_aus(bild, platz)))
        infos.append((platz["id"], platz["label"]))
    return ausschnitte, infos


def lese_csv(pfad):
    with open(pfad, encoding="utf-8") as datei:
        return list(csv.DictReader(datei))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mengen", nargs="+", default=STANDARD_MENGEN)
    parser.add_argument("--limit", type=int, default=None,
                        help="nur die ersten N Bilder pro Menge (zum Testen)")
    args = parser.parse_args()

    bilder = lese_csv(ERGEBNIS_ORDNER / "datenuebersicht_bilder.csv")
    menge_von = {(z["parkplatz"], z["tag"]): z["menge"]
                 for z in lese_csv(ERGEBNIS_ORDNER / "split_tage.csv")}
    CACHE_ORDNER.mkdir(parents=True, exist_ok=True)

    for menge in args.mengen:
        start = time.time()
        zeilen = sorted((b for b in bilder if menge_von[(b["parkplatz"], b["tag"])] == menge),
                        key=lambda b: (b["parkplatz"], b["tag"], b["datei"]))
        if args.limit:
            zeilen = zeilen[:args.limit]
        anzahl = sum(int(z["frei"]) + int(z["belegt"]) for z in zeilen)

        # Speicher direkt auf der Festplatte anlegen, damit nicht alles in den RAM muss
        speicher = np.lib.format.open_memmap(
            CACHE_ORDNER / f"{menge}_bilder.npy", mode="w+", dtype=np.uint8,
            shape=(anzahl, EINGABE_GROESSE, EINGABE_GROESSE, 3))

        position = 0
        with open(CACHE_ORDNER / f"{menge}_info.csv", "w", newline="", encoding="utf-8") as datei, \
                Pool() as pool:
            schreiber = csv.writer(datei)
            schreiber.writerow(["index", "parkplatz", "wetter", "tag", "datei", "platz_id", "label"])
            # imap behält die Reihenfolge bei, das Ergebnis ist also reproduzierbar
            for zeile, (ausschnitte, infos) in zip(zeilen, pool.imap(verarbeite_bild, zeilen, chunksize=8)):
                for ausschnitt, (platz_id, label) in zip(ausschnitte, infos):
                    speicher[position] = ausschnitt
                    schreiber.writerow([position, zeile["parkplatz"], zeile["wetter"],
                                        zeile["tag"], zeile["datei"], platz_id, label])
                    position += 1
        speicher.flush()
        assert position == anzahl, f"{menge}: erwartet {anzahl}, geschrieben {position}"
        print(f"{menge}: {anzahl} Ausschnitte aus {len(zeilen)} Bildern "
              f"in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
