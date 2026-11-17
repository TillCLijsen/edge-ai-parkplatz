"""Fehleranalyse des Hauptmodells (MobileNetV3-Small INT8) und Abbildungen für den Bericht.

1. Ausreisser: die Testbilder mit den meisten falsch erkannten Plätzen; die
   drei schlimmsten werden mit markierten Fehlern gespeichert.
2. Fehlklassifikationen: die Ausschnitte, bei denen das Modell am sichersten
   falsch lag, je 8 pro Fehlerrichtung.
3. Fehler nach Wetter: Fehler pro 10.000 Plätze, getrennt nach Richtung
   (frei als belegt erkannt / belegt als frei erkannt).
4. Genauigkeit gegen Latenz: alle gemessenen Varianten in einer Abbildung.

Aufruf:  python -m src.fehleranalyse
Ausgabe: ergebnisse/fehler_ausreisser.csv, ergebnisse/fehler_wetter.csv,
         ergebnisse/abbildungen/fehler_*.png, ergebnisse/abbildungen/genauigkeit_latenz.png
"""

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

matplotlib.use("Agg")  # Abbildungen nur als Datei, ohne Fenster
import matplotlib.pyplot as plt  # noqa: E402

from src.ausschnitte_vorberechnen import CACHE_ORDNER, PKLOT  # noqa: E402
from src.konfiguration import ERGEBNIS_ORDNER, ONNX_ORDNER, starte_protokoll  # noqa: E402
from src.messung import erstelle_sitzung  # noqa: E402
from src.vorverarbeitung import lese_plaetze, normalisiere  # noqa: E402

MODELL = "mobilenet_v3_small_seed43_int8"
ABBILDUNGS_ORDNER = ERGEBNIS_ORDNER / "abbildungen"
BLAU, ORANGE = "#2a78d6", "#eb6834"
TEXT, TEXT_HELL, GITTER = "#0b0b0b", "#52514e", "#e1e0d9"
RICHTUNG = {"fp": "frei, als belegt erkannt", "fn": "belegt, als frei erkannt"}


def vorhersagen():
    """Wahrscheinlichkeit „belegt“ für jeden Ausschnitt aus Test A und B."""
    sitzung = erstelle_sitzung(ONNX_ORDNER / f"{MODELL}.onnx", 4)
    tabellen = []
    for menge in ["test_a", "test_b"]:
        bilder = np.load(CACHE_ORDNER / f"{menge}_bilder.npy", mmap_mode="r")
        info = pd.read_csv(CACHE_ORDNER / f"{menge}_info.csv")
        logits = np.concatenate([
            sitzung.run(None, {"eingabe": normalisiere(np.array(bilder[i:i + 1024]))})[0]
            for i in range(0, len(bilder), 1024)])
        info["menge"] = menge
        info["p_belegt"] = 1 / (1 + np.exp(logits[:, 0] - logits[:, 1]))  # Softmax
        info["vorhersage"] = (info["p_belegt"] >= 0.5).astype(int)
        tabellen.append(info)
    daten = pd.concat(tabellen, ignore_index=True)
    daten["fp"] = ((daten["label"] == 0) & (daten["vorhersage"] == 1)).astype(int)
    daten["fn"] = ((daten["label"] == 1) & (daten["vorhersage"] == 0)).astype(int)
    return daten


def ausreisser(daten, anzahl=10):
    """Bilder mit den meisten falsch erkannten Plätzen."""
    daten = daten.assign(frei_echt=(daten["label"] == 0).astype(int),
                         frei_gezaehlt=(daten["vorhersage"] == 0).astype(int))
    pro_bild = (daten.groupby(["menge", "parkplatz", "wetter", "tag", "datei"])
                [["frei_echt", "frei_gezaehlt", "fp", "fn"]].sum().reset_index())
    pro_bild["fehler"] = pro_bild["frei_gezaehlt"] - pro_bild["frei_echt"]
    pro_bild["falsch_klassifiziert"] = pro_bild["fp"] + pro_bild["fn"]
    return pro_bild.sort_values(["falsch_klassifiziert", "datei"], ascending=[False, True]).head(anzahl)


def ecken(platz):
    """Eckpunkte des gedrehten Rechtecks eines Platzes (für die Zeichnung)."""
    (mx, my), (b, h), winkel = platz["mitte"], platz["groesse"], np.deg2rad(platz["winkel"])
    c, s = np.cos(winkel), np.sin(winkel)
    return [(mx + dx * c - dy * s, my + dx * s + dy * c)
            for dx, dy in [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]]


def zeichne_bild(zeile, daten, ziel):
    """Kamerabild mit allen Plätzen; falsch erkannte Plätze farbig und dick umrandet."""
    ordner = PKLOT / zeile["parkplatz"] / zeile["wetter"] / zeile["tag"]
    bild = Image.open(ordner / zeile["datei"]).convert("RGB")
    plaetze = lese_plaetze(ordner / zeile["datei"].replace(".jpg", ".xml"))
    teil = daten[(daten["parkplatz"] == zeile["parkplatz"]) & (daten["datei"] == zeile["datei"])]
    status = {int(z.platz_id): ("fp" if z.fp else "fn" if z.fn else "ok") for z in teil.itertuples()}
    zeichnung = ImageDraw.Draw(bild)
    for platz in plaetze:
        art = status.get(platz["id"], "ohne_label")
        farbe = {"fp": ORANGE, "fn": BLAU}.get(art, "#ffffff")
        zeichnung.polygon(ecken(platz), outline=farbe, width=4 if art in ("fp", "fn") else 1)
    bild.save(ziel)


def zeichne_fehlklassifikationen(daten, ziel, pro_richtung=8):
    """Die sichersten Fehler je Richtung als Ausschnitt-Raster."""
    caches = {m: np.load(CACHE_ORDNER / f"{m}_bilder.npy", mmap_mode="r") for m in ["test_a", "test_b"]}
    fig, achsen = plt.subplots(2, pro_richtung, figsize=(pro_richtung * 1.3, 3.6))
    for reihe, richtung in enumerate(["fp", "fn"]):
        fehler = daten[daten[richtung] == 1].copy()
        # Sicherheit des Modells in der falschen Klasse
        fehler["sicherheit"] = fehler["p_belegt"] if richtung == "fp" else 1 - fehler["p_belegt"]
        fehler = fehler.sort_values(["sicherheit", "index"], ascending=[False, True]).head(pro_richtung)
        for spalte in range(pro_richtung):
            achse = achsen[reihe, spalte]
            achse.axis("off")
            if spalte >= len(fehler):
                continue
            z = fehler.iloc[spalte]
            achse.imshow(caches[z["menge"]][int(z["index"])])
            achse.set_title(f"{z['parkplatz']}\n{z['wetter']}, {z['sicherheit']:.0%}",
                            fontsize=7, color=TEXT_HELL)
        achsen[reihe, 0].text(-0.15, 0.5, RICHTUNG[richtung], transform=achsen[reihe, 0].transAxes,
                              rotation=90, va="center", ha="right", fontsize=8, color=TEXT)
    fig.tight_layout()
    fig.savefig(ziel, dpi=200)
    plt.close(fig)


def fehler_nach_wetter(daten):
    """Fehler pro 10.000 Plätze nach Testmenge und Wetter, getrennt nach Richtung."""
    gruppe = daten.groupby(["menge", "wetter"])
    tabelle = pd.DataFrame({"plaetze": gruppe.size(),
                            "frei": gruppe["label"].apply(lambda s: int((s == 0).sum())),
                            "belegt": gruppe["label"].sum(), "fp": gruppe["fp"].sum(),
                            "fn": gruppe["fn"].sum()}).reset_index()
    tabelle["fehler_pro_10000"] = ((tabelle["fp"] + tabelle["fn"]) / tabelle["plaetze"] * 1e4).round(2)
    # Fehlerquote je Klasse, damit die unterschiedliche Belegung nicht verzerrt
    tabelle["fp_pro_10000_freie"] = (tabelle["fp"] / tabelle["frei"] * 1e4).round(2)
    tabelle["fn_pro_10000_belegte"] = (tabelle["fn"] / tabelle["belegt"] * 1e4).round(2)
    return tabelle


def zeichne_genauigkeit_latenz(ziel):
    """Fehler pro 10.000 Plätze (Test A und B zusammen) gegen Latenz (PUCPR, 1 Thread)."""
    genauigkeit = pd.read_csv(ERGEBNIS_ORDNER / "messung_genauigkeit.csv")
    genauigkeit = genauigkeit[genauigkeit["gruppe"] == "gesamt"]
    pytorch = pd.read_csv(ERGEBNIS_ORDNER / "evaluation_pytorch.csv")
    pytorch = pytorch[(pytorch["modell"] == "mobilenet_v3_small_seed43") & (pytorch["gruppe"] == "gesamt")]
    pytorch = pytorch.assign(modell="pytorch_mobilenet_v3_small_seed43")
    genauigkeit = pd.concat([genauigkeit, pytorch])
    latenz = pd.read_csv(ERGEBNIS_ORDNER / "messung_latenz.csv")
    latenz = latenz[(latenz["parkplatz"] == "PUCPR") & (latenz["threads"] == 1)]

    namen = {"pytorch_mobilenet_v3_small_seed43": "PyTorch FP32 (Referenz)",
             "mobilenet_v3_small_seed43_fp32": "MobileNet FP32",
             "mobilenet_v3_small_seed43_int8": "MobileNet INT8",
             "mobilenet_v3_small_seed43_pruned50_fp32": "MobileNet gepruned FP32",
             "mobilenet_v3_small_seed43_pruned50_int8": "MobileNet gepruned INT8",
             "resnet18_seed43_fp32": "ResNet-18 FP32",
             "resnet18_seed43_int8": "ResNet-18 INT8"}
    punkte = []
    for variante, name in namen.items():
        g = genauigkeit[genauigkeit["modell"] == variante]
        fehler = (g["fp"].sum() + g["fn"].sum()) / g["anzahl"].sum() * 1e4
        ms = float(latenz.loc[latenz["variante"] == variante, "gesamt_median_ms"].iloc[0])
        punkte.append((name, ms, fehler))

    # Versatz der Beschriftung, damit nahe Punkte lesbar bleiben
    versatz = {"MobileNet FP32": (7, -11), "MobileNet gepruned FP32": (7, 5),
               "MobileNet INT8": (7, 5), "MobileNet gepruned INT8": (7, -11)}
    fig, achse = plt.subplots(figsize=(7.5, 4))
    for name, ms, fehler in punkte:
        farbe = ORANGE if "ResNet" in name else BLAU
        achse.scatter(ms, fehler, s=60, color=farbe, edgecolors="white", linewidths=1.5, zorder=3)
        achse.annotate(name, (ms, fehler), textcoords="offset points",
                       xytext=versatz.get(name, (7, 4)), fontsize=8, color=TEXT)
    achse.set_xscale("log")
    achse.set_xlim(20, 1500)
    achse.set_xticks([20, 50, 100, 200, 500, 1000])
    achse.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    achse.minorticks_off()
    achse.set_xlabel("Latenz pro Kamerabild, PUCPR mit 100 Plätzen, 1 Thread (ms, log. Skala)",
                     color=TEXT_HELL)
    achse.set_ylabel("Fehler pro 10.000 Plätze\n(Test A und B)", color=TEXT_HELL)
    achse.set_ylim(0, max(f for _, _, f in punkte) * 1.25)
    achse.set_title("Genauigkeit gegen Latenz aller Varianten", loc="left", fontsize=11, color=TEXT)
    achse.grid(axis="y", color=GITTER, linewidth=0.8)
    for seite in ("top", "right"):
        achse.spines[seite].set_visible(False)
    achse.tick_params(colors=TEXT_HELL)
    fig.tight_layout()
    fig.savefig(ziel, dpi=200)
    plt.close(fig)
    return punkte


def main():
    starte_protokoll("fehleranalyse")
    ABBILDUNGS_ORDNER.mkdir(parents=True, exist_ok=True)
    daten = vorhersagen()
    print(f"{len(daten)} Ausschnitte, {int(daten['fp'].sum())} frei als belegt, "
          f"{int(daten['fn'].sum())} belegt als frei erkannt")

    # 1. Ausreisser
    schlimmste = ausreisser(daten)
    schlimmste.to_csv(ERGEBNIS_ORDNER / "fehler_ausreisser.csv", index=False)
    print("Bilder mit den meisten falsch erkannten Plätzen:")
    print(schlimmste.to_string(index=False))
    for nummer, (_, zeile) in enumerate(schlimmste.head(3).iterrows(), start=1):
        zeichne_bild(zeile, daten, ABBILDUNGS_ORDNER / f"fehler_bild_{nummer}.png")

    # 2. Fehlklassifikationen
    zeichne_fehlklassifikationen(daten, ABBILDUNGS_ORDNER / "fehler_beispiele.png")

    # 3. Wetter
    wetter = fehler_nach_wetter(daten)
    wetter.to_csv(ERGEBNIS_ORDNER / "fehler_wetter.csv", index=False)
    print("Fehler nach Wetter:")
    print(wetter.to_string(index=False))

    # 4. Genauigkeit gegen Latenz
    for name, ms, fehler in zeichne_genauigkeit_latenz(ABBILDUNGS_ORDNER / "genauigkeit_latenz.png"):
        print(f"{name:26s} {ms:7.1f} ms  {fehler:6.2f} Fehler pro 10.000 Plätze")
    print("Gespeichert: ergebnisse/fehler_ausreisser.csv, fehler_wetter.csv, "
          "abbildungen/fehler_bild_1-3.png, fehler_beispiele.png, genauigkeit_latenz.png")


if __name__ == "__main__":
    main()