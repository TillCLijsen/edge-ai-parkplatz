"""Prüft unsere Ausschnitte gegen die fertigen Ausschnitte von PKLot (PKLotSegmented).

Geprüft wird stichprobenartig:
1. Stimmt das Label aus der XML mit dem Ordner (Occupied/Empty) überein?
2. Zeigen unsere Ausschnitte denselben Bildinhalt (Korrelation der Grauwerte)?

Zusätzlich wird eine Vergleichsgrafik gespeichert (links unser, rechts PKLot).
"""

import csv
import random

import numpy as np
from PIL import Image

from src.konfiguration import DATEN_ORDNER, ERGEBNIS_ORDNER, SEED
from src.vorverarbeitung import lese_plaetze, schneide_platz_aus

STICHPROBE = 300
PKLOT = DATEN_ORDNER / "PKLot" / "PKLot"
SEGMENTIERT = DATEN_ORDNER / "PKLot" / "PKLotSegmented"
ORDNERNAME = {"PUCPR": "PUC", "UFPR04": "UFPR04", "UFPR05": "UFPR05"}


def als_grauwerte(bild):
    """Lange Seite senkrecht, 64×64, Grauwerte als Vektor."""
    if bild.width > bild.height:
        bild = bild.rotate(90, expand=True)
    return np.asarray(bild.convert("L").resize((64, 64)), dtype=np.float32).ravel()


def korrelation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    zufall = random.Random(SEED)
    with open(ERGEBNIS_ORDNER / "datenuebersicht_bilder.csv", encoding="utf-8") as datei:
        bilder = list(csv.DictReader(datei))

    ergebnisse, paare = [], []
    while len(ergebnisse) < STICHPROBE:
        b = zufall.choice(bilder)
        ordner = PKLOT / b["parkplatz"] / b["wetter"] / b["tag"]
        plaetze = [p for p in lese_plaetze(ordner / b["datei"].replace(".jpg", ".xml"))
                   if p["label"] is not None]
        if not plaetze:
            continue
        platz = zufall.choice(plaetze)
        name = f"{b['datei'][:-4]}#{platz['id']:03d}.jpg"
        seg_ordner = SEGMENTIERT / ORDNERNAME[b["parkplatz"]] / b["wetter"] / b["tag"]
        gefunden = [k for k in ("Empty", "Occupied") if (seg_ordner / k / name).exists()]
        if len(gefunden) != 1:
            ergebnisse.append({"gefunden": False})
            continue
        label_pklot = 1 if gefunden[0] == "Occupied" else 0
        referenz = Image.open(seg_ordner / gefunden[0] / name).convert("RGB")
        unser = schneide_platz_aus(Image.open(ordner / b["datei"]).convert("RGB"), platz)

        g_unser, g_ref = als_grauwerte(unser), als_grauwerte(referenz)
        # PKLot dreht auf die nächste Achse (0° oder 90°), wir richten die lange
        # Seite senkrecht aus. Deshalb alle vier Drehlagen prüfen (ohne Spiegelung).
        quadrat = g_unser.reshape(64, 64)
        r = max(korrelation(np.rot90(quadrat, k).ravel(), g_ref) for k in range(4))
        ergebnisse.append({"gefunden": True, "label_gleich": label_pklot == platz["label"],
                           "korrelation": r, "parkplatz": b["parkplatz"]})
        if len(paare) < 12:
            paare.append((unser, referenz))

    gefunden = [e for e in ergebnisse if e["gefunden"]]
    korr = np.array([e["korrelation"] for e in gefunden])
    print(f"Stichprobe: {len(ergebnisse)} Plätze, in PKLotSegmented gefunden: {len(gefunden)}")
    print(f"Label identisch: {sum(e['label_gleich'] for e in gefunden)} von {len(gefunden)}")
    print(f"Korrelation Median {np.median(korr):.3f}, Minimum {korr.min():.3f}, "
          f"Anteil > 0,8: {np.mean(korr > 0.8):.1%}")
    for name in ORDNERNAME:
        k = [e["korrelation"] for e in gefunden if e["parkplatz"] == name]
        if k:
            print(f"   {name}: n={len(k)}, Median {np.median(k):.3f}")

    # Vergleichsgrafik: jede Zeile = unser Ausschnitt | PKLot-Ausschnitt
    kachel = 96
    grafik = Image.new("RGB", (2 * kachel + 10, len(paare) * (kachel + 4)), "white")
    for i, (unser, referenz) in enumerate(paare):
        if referenz.width > referenz.height:
            referenz = referenz.rotate(90, expand=True)
        grafik.paste(unser.resize((kachel, kachel)), (0, i * (kachel + 4)))
        grafik.paste(referenz.resize((kachel, kachel)), (kachel + 10, i * (kachel + 4)))
    ziel = ERGEBNIS_ORDNER / "ausschnitte_vergleich.png"
    grafik.save(ziel)
    print("Vergleichsgrafik:", ziel)


if __name__ == "__main__":
    main()
