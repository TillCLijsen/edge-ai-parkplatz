"""Erstellt eine Übersicht über den PKLot-Datensatz (Bilder, Plätze, Labels).

Liest alle XML-Annotationen und schreibt eine Tabelle mit einer Zeile pro Bild
nach ergebnisse/datenuebersicht_bilder.csv. Zusätzlich werden Auffälligkeiten
(fehlende Labels, Bilder ohne XML usw.) in der Konsole ausgegeben.
"""

import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from src.konfiguration import DATEN_ORDNER, ERGEBNIS_ORDNER

PKLOT_ORDNER = DATEN_ORDNER / "PKLot" / "PKLot"


def lese_annotation(xml_pfad):
    """Liest eine XML-Datei und zählt freie, belegte und unbeschriftete Plätze."""
    wurzel = ET.parse(xml_pfad).getroot()
    frei = belegt = ohne_label = 0
    for platz in wurzel.iter("space"):
        status = platz.get("occupied")
        if status == "0":
            frei += 1
        elif status == "1":
            belegt += 1
        else:
            ohne_label += 1
    return frei, belegt, ohne_label


def main():
    zeilen = []
    probleme = Counter()

    # Ordnerstruktur: Parkplatz / Wetter / Tag / Bild.jpg + Bild.xml
    for bild_pfad in sorted(PKLOT_ORDNER.glob("*/*/*/*.jpg")):
        parkplatz, wetter, tag = bild_pfad.parts[-4:-1]
        xml_pfad = bild_pfad.with_suffix(".xml")
        if not xml_pfad.exists():
            probleme["Bild ohne XML"] += 1
            continue
        try:
            frei, belegt, ohne_label = lese_annotation(xml_pfad)
        except ET.ParseError:
            probleme["XML nicht lesbar"] += 1
            continue
        if ohne_label:
            probleme["Bilder mit unbeschrifteten Plätzen"] += 1
        zeilen.append({
            "datei": bild_pfad.name,
            "parkplatz": parkplatz,
            "wetter": wetter,
            "tag": tag,
            "uhrzeit": bild_pfad.stem.split("_", 1)[1],
            "plaetze": frei + belegt + ohne_label,
            "frei": frei,
            "belegt": belegt,
            "ohne_label": ohne_label,
            "bytes": bild_pfad.stat().st_size,
        })

    # XML-Dateien ohne zugehöriges Bild zählen
    for xml_pfad in PKLOT_ORDNER.glob("*/*/*/*.xml"):
        if not xml_pfad.with_suffix(".jpg").exists():
            probleme["XML ohne Bild"] += 1

    ERGEBNIS_ORDNER.mkdir(exist_ok=True)
    ziel = ERGEBNIS_ORDNER / "datenuebersicht_bilder.csv"
    with open(ziel, "w", newline="", encoding="utf-8") as datei:
        schreiber = csv.DictWriter(datei, fieldnames=list(zeilen[0].keys()))
        schreiber.writeheader()
        schreiber.writerows(zeilen)

    # Kurze Zusammenfassung pro Parkplatz
    pro_parkplatz = defaultdict(lambda: Counter())
    for z in zeilen:
        p = pro_parkplatz[z["parkplatz"]]
        p["bilder"] += 1
        p["frei"] += z["frei"]
        p["belegt"] += z["belegt"]
        p["ohne_label"] += z["ohne_label"]
        p["bytes"] += z["bytes"]
        p[("wetter", z["wetter"])] += 1
    for name, p in sorted(pro_parkplatz.items()):
        tage = {z["tag"] for z in zeilen if z["parkplatz"] == name}
        gelabelt = p["frei"] + p["belegt"]
        print(f"{name}: {p['bilder']} Bilder, {len(tage)} Tage, "
              f"{gelabelt} Plätze gelabelt (belegt {p['belegt'] / gelabelt:.1%}), "
              f"{p['ohne_label']} ohne Label, "
              f"Ø JPEG {p['bytes'] / p['bilder'] / 1024:.0f} KB")
        wetter = {k[1]: v for k, v in p.items() if isinstance(k, tuple)}
        print(f"   Wetter (Bilder): {wetter}")
    print("Auffälligkeiten:", dict(probleme))
    print("Gespeichert:", ziel)


if __name__ == "__main__":
    main()
