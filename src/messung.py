"""Misst alle Modellvarianten auf dem Zielgerät (MacBook mit Apple M2, nur CPU).

Phase 1  Grösse der Modelldatei (unkomprimiert und gzip-komprimiert)
Phase 2  Genauigkeit auf Test A und Test B mit ONNX Runtime
         (gleiche Kennzahlen wie src/evaluation.py, siehe src/metriken.py)
Phase 3  Latenz pro Kamerabild mit 1 und 4 Threads, jede Messung in einem
         eigenen Prozess: JPEG dekodieren + Plätze ausschneiden/vorverarbeiten
         + Inferenz + JSON-Meldung. Dazu der Spitzen-Arbeitsspeicher.

Referenz für die Latenz: dasselbe MobileNet unoptimiert in PyTorch auf der CPU.

Aufruf:  python -m src.messung
         python -m src.messung --nur-latenz
"""

import argparse
import csv
import gzip
import io
import json
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

from src.ausschnitte_vorberechnen import CACHE_ORDNER, PKLOT
from src.konfiguration import ERGEBNIS_ORDNER, ONNX_ORDNER, SEED
from src.metriken import werte_aus
from src.vorverarbeitung import bereite_bild_vor, lese_plaetze, normalisiere

THREADS = [1, 4]
BILDER_PRO_PARKPLATZ = 30
AUFWAERMEN = 10
RUNDEN = 3
PAUSE_SEKUNDEN = 20  # MacBook Air hat keinen Lüfter: zwischen den Läufen abkühlen lassen
PYTORCH_REFERENZ = "pytorch_mobilenet_v3_small_seed43"
TESTMENGE_VON = {"UFPR04": "test_a", "UFPR05": "test_a", "PUCPR": "test_b"}


def lese_csv(pfad):
    with open(pfad, encoding="utf-8") as datei:
        return list(csv.DictReader(datei))


def erstelle_sitzung(pfad, threads):
    """ONNX-Runtime-Sitzung nur auf der CPU mit fester Thread-Anzahl."""
    optionen = ort.SessionOptions()
    optionen.intra_op_num_threads = threads
    optionen.inter_op_num_threads = 1
    optionen.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    optionen.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(pfad), optionen, providers=["CPUExecutionProvider"])


def waehle_messbilder():
    """30 zufällige Testbilder pro Parkplatz, JPEG-Bytes vorab im Speicher.

    Die Platzkoordinaten werden vorab gelesen: Im Betrieb sind sie feste
    Konfiguration der Kamera und müssen nicht pro Bild geladen werden.
    """
    menge_von = {(z["parkplatz"], z["tag"]): z["menge"]
                 for z in lese_csv(ERGEBNIS_ORDNER / "split_tage.csv")}
    bilder = lese_csv(ERGEBNIS_ORDNER / "datenuebersicht_bilder.csv")
    zufall = np.random.default_rng(SEED)
    auswahl = []
    for parkplatz, menge in TESTMENGE_VON.items():
        kandidaten = [b for b in bilder if b["parkplatz"] == parkplatz
                      and menge_von[(parkplatz, b["tag"])] == menge]
        for i in sorted(zufall.choice(len(kandidaten), BILDER_PRO_PARKPLATZ, replace=False)):
            b = kandidaten[i]
            ordner = PKLOT / parkplatz / b["wetter"] / b["tag"]
            auswahl.append({
                "parkplatz": parkplatz,
                "jpeg": (ordner / b["datei"]).read_bytes(),
                "plaetze": lese_plaetze(ordner / b["datei"].replace(".jpg", ".xml")),
            })
    return auswahl


def einzelmessung(variante, threads):
    """Latenz und Arbeitsspeicher einer Variante (läuft in eigenem Prozess)."""
    ram_basis = psutil.Process().memory_info().rss
    messbilder = waehle_messbilder()

    if variante == PYTORCH_REFERENZ:
        import torch
        from src.export_onnx import lade_pytorch_modell
        torch.set_num_threads(threads)
        modell = lade_pytorch_modell(variante.removeprefix("pytorch_"))

        def inferenz(x):
            with torch.no_grad():
                return modell(torch.from_numpy(x)).numpy()
    else:
        sitzung = erstelle_sitzung(ONNX_ORDNER / f"{variante}.onnx", threads)

        def inferenz(x):
            return sitzung.run(None, {"eingabe": x})[0]

    def verarbeite(eintrag):
        start = time.perf_counter()
        bild = Image.open(io.BytesIO(eintrag["jpeg"]))
        x = bereite_bild_vor(bild, eintrag["plaetze"])
        nach_vorverarbeitung = time.perf_counter()
        ausgabe = inferenz(x)
        nach_inferenz = time.perf_counter()
        status = ausgabe.argmax(axis=1).tolist()
        meldung = json.dumps({"zeit": time.time(), "frei": status.count(0),
                              "status": status}).encode()
        ende = time.perf_counter()
        return (nach_vorverarbeitung - start, nach_inferenz - nach_vorverarbeitung,
                ende - start, len(meldung))

    for eintrag in messbilder[:AUFWAERMEN]:
        verarbeite(eintrag)

    messwerte = {p: [] for p in TESTMENGE_VON}
    for _ in range(RUNDEN):
        for eintrag in messbilder:
            messwerte[eintrag["parkplatz"]].append(verarbeite(eintrag))

    spitze = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        spitze *= 1024  # Linux meldet Kilobyte, macOS Bytes

    zeilen = []
    for parkplatz, werte in messwerte.items():
        w = np.array(werte)
        jpeg = np.mean([len(e["jpeg"]) for e in messbilder if e["parkplatz"] == parkplatz])
        zeilen.append({
            "variante": variante, "threads": threads, "parkplatz": parkplatz,
            "plaetze_pro_bild": len(next(e for e in messbilder if e["parkplatz"] == parkplatz)["plaetze"]),
            "messungen": len(w),
            "vorverarbeitung_median_ms": round(np.median(w[:, 0]) * 1000, 2),
            "inferenz_median_ms": round(np.median(w[:, 1]) * 1000, 2),
            "inferenz_p95_ms": round(np.percentile(w[:, 1], 95) * 1000, 2),
            "gesamt_median_ms": round(np.median(w[:, 2]) * 1000, 2),
            "gesamt_p95_ms": round(np.percentile(w[:, 2], 95) * 1000, 2),
            "bilder_pro_sekunde": round(1 / np.median(w[:, 2]), 1),
            "jpeg_kb": round(jpeg / 1024, 1),
            "meldung_bytes": int(np.median(w[:, 3])),
            "ram_basis_mb": round(ram_basis / 1e6, 1),
            "ram_spitze_mb": round(spitze / 1e6, 1),
        })
    print("ERGEBNIS:" + json.dumps(zeilen))


def genauigkeit(variante):
    """Kennzahlen der ONNX-Variante auf Test A und B (4 Threads, Stapel à 1024)."""
    sitzung = erstelle_sitzung(ONNX_ORDNER / f"{variante}.onnx", 4)
    zufall = np.random.default_rng(SEED)
    zeilen = []
    for menge in ["test_a", "test_b"]:
        bilder = np.load(CACHE_ORDNER / f"{menge}_bilder.npy", mmap_mode="r")
        wahrscheinlichkeiten = []
        for i in range(0, len(bilder), 1024):
            logits = sitzung.run(None, {"eingabe": normalisiere(np.array(bilder[i:i + 1024]))})[0]
            wahrscheinlichkeiten.append(1 / (1 + np.exp(logits[:, 0] - logits[:, 1])))  # Softmax
        zeilen += werte_aus(variante, menge, np.concatenate(wahrscheinlichkeiten), zufall)
    return zeilen


def messaufbau():
    """Hält Hardware und Software fest (für den Bericht)."""
    info = {"plattform": platform.platform(), "python": platform.python_version(),
            "onnxruntime": ort.__version__, "numpy": np.__version__,
            "kerne_physisch": psutil.cpu_count(logical=False),
            "kerne_logisch": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}
    if sys.platform == "darwin":
        for schluessel, befehl in [("chip", "machdep.cpu.brand_string"), ("modell", "hw.model")]:
            info[schluessel] = subprocess.run(["sysctl", "-n", befehl], capture_output=True,
                                              text=True).stdout.strip()
    akku = psutil.sensors_battery()
    info["netzteil_angeschlossen"] = akku.power_plugged if akku else None
    return info


def schreibe_csv(zeilen, dateiname):
    with open(ERGEBNIS_ORDNER / dateiname, "w", newline="", encoding="utf-8") as datei:
        schreiber = csv.DictWriter(datei, fieldnames=list(zeilen[0].keys()))
        schreiber.writeheader()
        schreiber.writerows(zeilen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--einzel", help="interne Nutzung: eine Latenzmessung")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--varianten", nargs="+")
    parser.add_argument("--nur-latenz", action="store_true")
    args = parser.parse_args()

    if args.einzel:
        einzelmessung(args.einzel, args.threads)
        return

    varianten = args.varianten or sorted(p.stem for p in ONNX_ORDNER.glob("*.onnx")
                                         if "vorbereitet" not in p.stem)
    aufbau = messaufbau()
    print("Messaufbau:", aufbau)
    if not aufbau.get("netzteil_angeschlossen", True):
        print("Hinweis: Netzteil anschliessen, im Akkubetrieb kann der Mac drosseln.")
    with open(ERGEBNIS_ORDNER / "messaufbau.json", "w", encoding="utf-8") as datei:
        json.dump(aufbau, datei, indent=2, ensure_ascii=False)

    # Phase 1: Grösse
    groessen = []
    for v in varianten:
        inhalt = (ONNX_ORDNER / f"{v}.onnx").read_bytes()
        groessen.append({"variante": v, "mb": round(len(inhalt) / 1e6, 2),
                         "mb_gzip": round(len(gzip.compress(inhalt, 9)) / 1e6, 2)})
    schreibe_csv(groessen, "messung_groesse.csv")
    for g in groessen:
        print(f"{g['variante']:42} {g['mb']:7.2f} MB   gzip {g['mb_gzip']:7.2f} MB")

    # Phase 2: Genauigkeit
    if not args.nur_latenz:
        alle = []
        for v in varianten:
            zeilen = genauigkeit(v)
            alle += zeilen
            for z in (z for z in zeilen if z["gruppe"] == "gesamt"):
                print(f"{v:42} {z['menge']}: BalAcc {z['balanced_accuracy']:.4f} "
                      f"[{z['bal_acc_ki_unten']:.4f}; {z['bal_acc_ki_oben']:.4f}]  "
                      f"Fehler/10.000 {z['fehler_pro_10000']}")
        schreibe_csv(alle, "messung_genauigkeit.csv")

    # Phase 3: Latenz und Arbeitsspeicher, jede Messung in eigenem Prozess
    latenzen = []
    for v in [PYTORCH_REFERENZ] + varianten:
        for threads in THREADS:
            time.sleep(PAUSE_SEKUNDEN)
            lauf = subprocess.run([sys.executable, "-m", "src.messung", "--einzel", v,
                                   "--threads", str(threads)],
                                  capture_output=True, text=True, check=True)
            zeile = next(z for z in lauf.stdout.splitlines() if z.startswith("ERGEBNIS:"))
            ergebnis = json.loads(zeile.removeprefix("ERGEBNIS:"))
            latenzen += ergebnis
            pucpr = next(e for e in ergebnis if e["parkplatz"] == "PUCPR")
            print(f"{v:42} {threads} Thread(s): PUCPR (100 Plätze) gesamt "
                  f"{pucpr['gesamt_median_ms']:7.1f} ms (P95 {pucpr['gesamt_p95_ms']:.1f}), "
                  f"Inferenz {pucpr['inferenz_median_ms']:.1f} ms, RAM {pucpr['ram_spitze_mb']} MB")
    schreibe_csv(latenzen, "messung_latenz.csv")
    print("Gespeichert: ergebnisse/messaufbau.json, messung_groesse.csv, "
          "messung_genauigkeit.csv, messung_latenz.csv")


if __name__ == "__main__":
    main()
