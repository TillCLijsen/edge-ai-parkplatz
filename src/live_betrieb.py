"""Simulierter Live-Betrieb: ein Testtag pro Parkplatz als Kamera-Stream.

Teil 1 (Stream): Pro Parkplatz wird ein zufällig gewählter Testtag (Seed 42)
Bild für Bild in der Reihenfolge der Aufnahmezeit abgespielt. Pro Bild läuft
die Kette wie auf dem Gerät: JPEG dekodieren, Plätze ausschneiden,
MobileNetV3-Small INT8 mit 1 Thread, JSON-Meldung. Die Meldungen werden in
eine Datei geschrieben; sie ersetzt die Übertragung an einen Server.

Teil 2 (Zählung): Für den Betreiber zählt die Zahl freier Plätze. Verglichen
wird pro Bild die gezählte mit der echten Zahl freier Plätze, einmal für die
abgespielten Tage und einmal für alle Testbilder (aus den vorberechneten
Ausschnitten). Plätze ohne Annotation zählen beim Vergleich nicht mit (E9).

Aufruf:  python -m src.live_betrieb
Ausgabe: ergebnisse/live_meldungen_<parkplatz>.jsonl, ergebnisse/live_bilder.csv,
         ergebnisse/live_zaehlung.csv, ergebnisse/abbildungen/live_<parkplatz>.png
"""

import io
import json
import time

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")  # Abbildungen nur als Datei, ohne Fenster
import matplotlib.pyplot as plt  # noqa: E402

from src.ausschnitte_vorberechnen import CACHE_ORDNER, PKLOT  # noqa: E402
from src.konfiguration import ERGEBNIS_ORDNER, ONNX_ORDNER, SEED, starte_protokoll  # noqa: E402
from src.messung import TESTMENGE_VON, erstelle_sitzung, lese_csv, schreibe_csv  # noqa: E402
from src.vorverarbeitung import bereite_bild_vor, lese_plaetze, normalisiere  # noqa: E402

MODELL = "mobilenet_v3_small_seed43_int8"
THREADS = 1  # laut Energiemessung die sparsamste Einstellung
ABBILDUNGS_ORDNER = ERGEBNIS_ORDNER / "abbildungen"
FARBE_ECHT = "#2a78d6"
FARBE_GEZAEHLT = "#eb6834"


def waehle_tage():
    """Pro Parkplatz ein zufälliger Tag aus seiner Testmenge (Seed 42)."""
    split = lese_csv(ERGEBNIS_ORDNER / "split_tage.csv")
    zufall = np.random.default_rng(SEED)
    tage = {}
    for parkplatz, menge in TESTMENGE_VON.items():
        kandidaten = sorted(z["tag"] for z in split
                            if z["parkplatz"] == parkplatz and z["menge"] == menge)
        tage[parkplatz] = kandidaten[zufall.integers(len(kandidaten))]
    return tage


def spiele_tag_ab(parkplatz, tag, sitzung, uebersicht):
    """Spielt alle Bilder eines Tages ab und schreibt die Meldungen in eine Datei."""
    bilder = sorted((b for b in uebersicht if b["parkplatz"] == parkplatz and b["tag"] == tag),
                    key=lambda b: b["uhrzeit"])
    zeilen = []
    with open(ERGEBNIS_ORDNER / f"live_meldungen_{parkplatz}.jsonl", "w",
              encoding="utf-8") as meldungen:
        for b in bilder:
            ordner = PKLOT / parkplatz / b["wetter"] / tag
            plaetze = lese_plaetze(ordner / b["datei"].replace(".jpg", ".xml"))
            jpeg = (ordner / b["datei"]).read_bytes()  # kommt im Betrieb von der Kamera

            start = time.perf_counter()
            x = bereite_bild_vor(Image.open(io.BytesIO(jpeg)), plaetze)
            status = sitzung.run(None, {"eingabe": x})[0].argmax(axis=1).tolist()
            aufnahme = f"{tag} {b['uhrzeit'].replace('_', ':')}"
            meldung = json.dumps({"parkplatz": parkplatz, "aufnahme": aufnahme,
                                  "plaetze": len(status), "frei": status.count(0),
                                  "status": status})
            dauer_ms = (time.perf_counter() - start) * 1000
            meldungen.write(meldung + "\n")

            # Vergleich nur über Plätze mit Annotation
            mit_label = [(p["label"], s) for p, s in zip(plaetze, status) if p["label"] is not None]
            frei_echt = sum(1 for label, _ in mit_label if label == 0)
            frei_gezaehlt = sum(1 for _, s in mit_label if s == 0)
            zeilen.append({"parkplatz": parkplatz, "tag": tag, "uhrzeit": b["uhrzeit"],
                           "wetter": b["wetter"], "plaetze": len(plaetze),
                           "plaetze_mit_label": len(mit_label),
                           "frei_echt": frei_echt, "frei_gezaehlt": frei_gezaehlt,
                           "fehler": frei_gezaehlt - frei_echt,
                           "frei_gemeldet_alle_plaetze": status.count(0),
                           "latenz_ms": round(dauer_ms, 2), "meldung_bytes": len(meldung)})
    return zeilen


def zaehlkennzahlen(fehler):
    """Kennzahlen zum Zählfehler (gezählt minus echt) über mehrere Bilder."""
    fehler = np.asarray(fehler)
    return {"bilder": len(fehler),
            "mittlerer_abs_fehler": round(float(np.abs(fehler).mean()), 3),
            "mittlerer_fehler": round(float(fehler.mean()), 3),
            "anteil_exakt": round(float((fehler == 0).mean()), 4),
            "anteil_hoechstens_1": round(float((np.abs(fehler) <= 1).mean()), 4),
            "max_abs_fehler": int(np.abs(fehler).max())}


def zaehlung_alle_testbilder():
    """Zählfehler pro Bild über alle Testbilder (vorberechnete Ausschnitte, 4 Threads)."""
    sitzung = erstelle_sitzung(ONNX_ORDNER / f"{MODELL}.onnx", 4)
    pro_bild = []
    for menge in ["test_a", "test_b"]:
        bilder = np.load(CACHE_ORDNER / f"{menge}_bilder.npy", mmap_mode="r")
        info = pd.read_csv(CACHE_ORDNER / f"{menge}_info.csv")
        vorhersage = np.concatenate([
            sitzung.run(None, {"eingabe": normalisiere(np.array(bilder[i:i + 1024]))})[0].argmax(1)
            for i in range(0, len(bilder), 1024)])
        info["frei_echt"] = (info["label"] == 0).astype(int)
        info["frei_gezaehlt"] = (vorhersage == 0).astype(int)
        pro_bild.append(info.groupby(["parkplatz", "tag", "datei"])[["frei_echt", "frei_gezaehlt"]]
                        .sum().reset_index())
    pro_bild = pd.concat(pro_bild)
    pro_bild["fehler"] = pro_bild["frei_gezaehlt"] - pro_bild["frei_echt"]
    return pro_bild


def zeichne_tag(zeilen, parkplatz, tag):
    """Freie Plätze über den Tag: gezählt gegen echt."""
    stunden = [int(z["uhrzeit"][:2]) + int(z["uhrzeit"][3:5]) / 60 for z in zeilen]
    fig, achse = plt.subplots(figsize=(8, 3.5))
    achse.plot(stunden, [z["frei_echt"] for z in zeilen], color=FARBE_ECHT, linewidth=2,
               label="echt (Annotation)")
    achse.plot(stunden, [z["frei_gezaehlt"] for z in zeilen], color=FARBE_GEZAEHLT, linewidth=2,
               linestyle="--", label="gezählt (MobileNet INT8)")
    # Bilder mit falscher Zählung hervorheben
    abweichung = [(s, z["frei_gezaehlt"]) for s, z in zip(stunden, zeilen) if z["fehler"] != 0]
    if abweichung:
        achse.scatter(*zip(*abweichung), color=FARBE_GEZAEHLT, s=30, zorder=3,
                      edgecolors="white", linewidths=1,
                      label=f"Abweichung ({len(abweichung)} Bilder)")
    achse.set_title(f"{parkplatz}, {tag}: freie Plätze mit Annotation", loc="left",
                    fontsize=11, color="#0b0b0b")
    achse.set_xlabel("Uhrzeit (h)", color="#52514e")
    achse.set_ylabel("freie Plätze", color="#52514e")
    achse.set_ylim(bottom=0)
    achse.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    for seite in ("top", "right"):
        achse.spines[seite].set_visible(False)
    achse.tick_params(colors="#52514e")
    achse.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(ABBILDUNGS_ORDNER / f"live_{parkplatz}.png", dpi=200)
    plt.close(fig)


def main():
    starte_protokoll("live_betrieb")
    ABBILDUNGS_ORDNER.mkdir(parents=True, exist_ok=True)
    uebersicht = lese_csv(ERGEBNIS_ORDNER / "datenuebersicht_bilder.csv")
    sitzung = erstelle_sitzung(ONNX_ORDNER / f"{MODELL}.onnx", THREADS)

    # Teil 1: Stream
    alle_zeilen, zusammenfassung = [], []
    for parkplatz, tag in waehle_tage().items():
        zeilen = spiele_tag_ab(parkplatz, tag, sitzung, uebersicht)
        alle_zeilen += zeilen
        zeichne_tag(zeilen, parkplatz, tag)
        latenz = np.array([z["latenz_ms"] for z in zeilen])
        sekunden = [int(z["uhrzeit"][:2]) * 3600 + int(z["uhrzeit"][3:5]) * 60
                    + int(z["uhrzeit"][6:8]) for z in zeilen]
        abstand = float(np.median(np.diff(sekunden))) if len(sekunden) > 1 else float("nan")
        kennzahlen = zaehlkennzahlen([z["fehler"] for z in zeilen])
        zusammenfassung.append({"quelle": "live_tag", "parkplatz": parkplatz, "tag": tag,
                                **kennzahlen,
                                "latenz_median_ms": round(float(np.median(latenz)), 2),
                                "latenz_max_ms": round(float(latenz.max()), 2),
                                "bildabstand_median_s": round(abstand, 1)})
        print(f"{parkplatz} {tag}: {len(zeilen)} Bilder, Zählfehler mittel "
              f"{kennzahlen['mittlerer_abs_fehler']}, exakt {kennzahlen['anteil_exakt']:.1%}, "
              f"max {kennzahlen['max_abs_fehler']}; Latenz Median {np.median(latenz):.1f} ms, "
              f"max {latenz.max():.1f} ms; Bildabstand im Datensatz {abstand / 60:.1f} min")
    schreibe_csv(alle_zeilen, "live_bilder.csv")

    # Teil 2: alle Testbilder
    pro_bild = zaehlung_alle_testbilder()
    for parkplatz, gruppe in pro_bild.groupby("parkplatz"):
        kennzahlen = zaehlkennzahlen(gruppe["fehler"])
        zusammenfassung.append({"quelle": "alle_testbilder", "parkplatz": parkplatz, "tag": "alle",
                                **kennzahlen, "latenz_median_ms": "", "latenz_max_ms": "",
                                "bildabstand_median_s": ""})
        print(f"{parkplatz} alle Testbilder: {kennzahlen['bilder']} Bilder, Zählfehler mittel "
              f"{kennzahlen['mittlerer_abs_fehler']}, exakt {kennzahlen['anteil_exakt']:.1%}, "
              f"höchstens 1 daneben {kennzahlen['anteil_hoechstens_1']:.1%}, "
              f"max {kennzahlen['max_abs_fehler']}")
    schreibe_csv(zusammenfassung, "live_zaehlung.csv")
    print("Gespeichert: ergebnisse/live_meldungen_*.jsonl, live_bilder.csv, live_zaehlung.csv, "
          "abbildungen/live_*.png")


if __name__ == "__main__":
    main()