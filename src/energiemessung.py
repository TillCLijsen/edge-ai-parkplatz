"""Energiemessung mit powermetrics (macOS, Apple Silicon).

Ablauf: powermetrics läuft neben der Messung und schreibt alle 500 ms die
geschätzte Leistungsaufnahme der CPU (cpu_power, in mW) in eine Datei.
Währenddessen laufen nacheinander:
  1. 30 s Leerlauf (Grundlast des Macs)
  2. je Variante und Thread-Anzahl 30 s Dauerbetrieb mit den PUCPR-Testbildern
     (100 Plätze, ganze Kette wie in src/messung.py), danach 20 s Pause
  3. nochmals 30 s Leerlauf
Eine Probe zählt nur, wenn sie vollständig in einer Phase liegt
(je 1 s Rand am Anfang und Ende wird verworfen).

Energie pro Bild:
  brutto = mittlere CPU-Leistung unter Last / Bilder pro Sekunde
  netto  = (CPU-Leistung unter Last - CPU-Leistung im Leerlauf) / Bilder pro Sekunde
Einschränkung: powermetrics liefert eine Schätzung aus Apples Energiemodell,
keine elektrische Messung.

powermetrics braucht Administratorrechte: macOS fragt beim Start einmal
nach dem Passwort. Weil macOS den Schreibtisch-Ordner auch vor diesem
Prozess schützt, schreibt powermetrics zuerst nach /tmp.

Aufruf:  python -m src.energiemessung
Ausgabe: ergebnisse/energie_phasen.csv, ergebnisse/energie_proben.csv,
         Rohdaten in daten/energie/
"""

import argparse
import io
import json
import math
import os
import plistlib
import shutil
import subprocess
import sys
import time
from datetime import timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.konfiguration import DATEN_ORDNER, ONNX_ORDNER, PROJEKT_ORDNER, starte_protokoll
from src.messung import (AUFWAERMEN, PYTORCH_REFERENZ, erstelle_sitzung, messaufbau,
                         schreibe_csv, waehle_messbilder)
from src.vorverarbeitung import bereite_bild_vor

STANDARD_VARIANTEN = [PYTORCH_REFERENZ, "mobilenet_v3_small_seed43_fp32",
                      "mobilenet_v3_small_seed43_int8", "resnet18_seed43_fp32",
                      "resnet18_seed43_int8"]
THREADS = [1, 4]
INTERVALL_MS = 500
LEERLAUF_SEKUNDEN = 30
LAST_SEKUNDEN = 30
PAUSE_SEKUNDEN = 20
RAND_SEKUNDEN = 1
ROH_ORDNER = DATEN_ORDNER / "energie"
TEMP_ORDNER = Path("/tmp")  # nicht vom macOS-Datenschutz betroffen


def lade_inferenz(variante, threads):
    """Gibt eine Funktion zurück, die einen Stapel Ausschnitte klassifiziert."""
    if variante == PYTORCH_REFERENZ:
        import torch
        from src.export_onnx import lade_pytorch_modell
        torch.set_num_threads(threads)
        modell = lade_pytorch_modell(variante.removeprefix("pytorch_"))

        def inferenz(x):
            with torch.no_grad():
                return modell(torch.from_numpy(x)).numpy()
        return inferenz
    sitzung = erstelle_sitzung(ONNX_ORDNER / f"{variante}.onnx", threads)
    return lambda x: sitzung.run(None, {"eingabe": x})[0]


def last(variante, threads, sekunden):
    """Dauerbetrieb einer Variante (läuft in eigenem Prozess)."""
    messbilder = [b for b in waehle_messbilder() if b["parkplatz"] == "PUCPR"]
    inferenz = lade_inferenz(variante, threads)

    def verarbeite(eintrag):
        bild = Image.open(io.BytesIO(eintrag["jpeg"]))
        status = inferenz(bereite_bild_vor(bild, eintrag["plaetze"])).argmax(axis=1).tolist()
        return json.dumps({"zeit": time.time(), "frei": status.count(0), "status": status})

    for eintrag in messbilder[:AUFWAERMEN]:
        verarbeite(eintrag)
    start = time.time()
    bilder = 0
    while time.time() - start < sekunden:
        verarbeite(messbilder[bilder % len(messbilder)])
        bilder += 1
    print("ERGEBNIS:" + json.dumps({"start": start, "ende": time.time(), "bilder": bilder}))


def starte_powermetrics(datei, fehlerdatei, anzahl):
    """Startet powermetrics mit Administratorrechten (Passwortfenster).

    osascript kehrt erst zurück, wenn powermetrics fertig ist. Darum läuft es
    als eigener Prozess neben der Messung.
    """
    befehl = (f"/usr/bin/powermetrics --samplers cpu_power -i {INTERVALL_MS} -n {anzahl} "
              f"-f plist -o '{datei}' 2> '{fehlerdatei}'")
    return subprocess.Popen(["osascript", "-e",
                             f'do shell script "{befehl}" with administrator privileges'])


def lese_proben(datei):
    """Liest alle vollständigen Proben aus der plist-Datei (durch Nullbytes getrennt)."""
    proben = []
    for teil in datei.read_bytes().split(b"\x00"):
        teil = teil.strip()
        if not teil:
            continue
        try:
            probe = plistlib.loads(teil)
        except Exception:
            continue  # letzte Probe ist evtl. noch unvollständig
        if "timestamp" not in probe or "cpu_power" not in probe.get("processor", {}):
            print("Unerwartetes Format, vorhandene Schlüssel:", sorted(probe),
                  sorted(probe.get("processor", {})))
            raise KeyError("timestamp oder processor/cpu_power fehlt")
        # plistlib liefert die Zeit ohne Zeitzone, gemeint ist UTC
        ende = probe["timestamp"].replace(tzinfo=timezone.utc).timestamp()
        dauer = probe.get("elapsed_ns", INTERVALL_MS * 1e6) / 1e9
        proben.append({"anfang": ende - dauer, "ende": ende,
                       "cpu_mw": probe["processor"]["cpu_power"],
                       "cpu_gpu_ane_mw": probe["processor"].get("combined_power")})
    return proben


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", help="interne Nutzung: Dauerbetrieb einer Variante")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--varianten", nargs="+", default=STANDARD_VARIANTEN)
    args = parser.parse_args()

    if args.last:
        last(args.last, args.threads, LAST_SEKUNDEN)
        return

    starte_protokoll("energiemessung")
    if sys.platform != "darwin":
        sys.exit("powermetrics gibt es nur auf macOS.")
    aufbau = messaufbau()
    print("Messaufbau:", aufbau)
    if not aufbau["netzteil_angeschlossen"]:
        print("Hinweis: Netzteil ist nicht angeschlossen.")
    # Mac wach halten, solange die Messung läuft
    subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    plan = [(v, t) for v in args.varianten for t in THREADS]
    geplant = 2 * LEERLAUF_SEKUNDEN + len(plan) * (LAST_SEKUNDEN + PAUSE_SEKUNDEN + 15) + 30
    zeitstempel = time.strftime("%Y%m%d_%H%M%S")
    roh = TEMP_ORDNER / f"powermetrics_{zeitstempel}.plist"
    fehler = TEMP_ORDNER / f"powermetrics_{zeitstempel}.fehler.txt"
    roh.touch()  # Datei gehört so dir und bleibt lesbar, auch wenn powermetrics als root schreibt
    fehler.touch()

    print(f"Bitte im macOS-Fenster das Passwort eingeben. Dauer ca. {geplant // 60} min.")
    powermetrics = starte_powermetrics(roh, fehler, math.ceil(geplant * 1000 / INTERVALL_MS))
    frist = time.time() + 120  # Zeit für die Passworteingabe
    while b"</plist>" not in roh.read_bytes():
        if powermetrics.poll() is not None or time.time() > frist:
            print("powermetrics schreibt keine Daten. Fehlermeldung:",
                  fehler.read_text() or "(keine)")
            sys.exit(1)
        time.sleep(0.5)
    print("powermetrics läuft.")

    phasen = []

    def leerlauf(name):
        start = time.time()
        time.sleep(LEERLAUF_SEKUNDEN)
        phasen.append({"phase": name, "variante": "-", "threads": 0,
                       "start": start, "ende": time.time(), "bilder": 0})
        print(f"{name}: fertig")

    leerlauf("leerlauf_vorher")
    for variante, threads in plan:
        lauf = subprocess.run([sys.executable, "-m", "src.energiemessung", "--last", variante,
                               "--threads", str(threads)],
                              capture_output=True, text=True, check=True, cwd=PROJEKT_ORDNER)
        zeile = next(z for z in lauf.stdout.splitlines() if z.startswith("ERGEBNIS:"))
        ergebnis = json.loads(zeile.removeprefix("ERGEBNIS:"))
        phasen.append({"phase": "last", "variante": variante, "threads": threads, **ergebnis})
        dauer = ergebnis["ende"] - ergebnis["start"]
        print(f"{variante:40s} {threads} Thread(s): {ergebnis['bilder'] / dauer:.1f} Bilder/s")
        time.sleep(PAUSE_SEKUNDEN)
    leerlauf("leerlauf_nachher")

    # Warten, bis powermetrics die letzte Phase vollständig geschrieben hat
    frist = time.time() + 120
    while True:
        proben = lese_proben(roh)
        if proben and proben[-1]["ende"] >= phasen[-1]["ende"]:
            break
        if time.time() > frist:
            print("Warnung: powermetrics hat die letzte Phase nicht vollständig erfasst.")
            break
        time.sleep(2)

    # Rohdaten in den Projektordner kopieren
    ROH_ORDNER.mkdir(parents=True, exist_ok=True)
    shutil.copy(roh, ROH_ORDNER / roh.name)

    # Proben den Phasen zuordnen
    for p in phasen:
        name = p["phase"] if p["phase"] != "last" else f"{p['variante']}_{p['threads']}t"
        drin = [q for q in proben if q["anfang"] >= p["start"] + RAND_SEKUNDEN
                and q["ende"] <= p["ende"] - RAND_SEKUNDEN]
        for q in drin:
            q["phase"] = name
        p["proben"] = len(drin)
        p["cpu_mw"] = float(np.mean([q["cpu_mw"] for q in drin])) if drin else float("nan")
        p["cpu_gpu_ane_mw"] = (float(np.mean([q["cpu_gpu_ane_mw"] for q in drin]))
                               if drin else float("nan"))
    leerlauf_mw = float(np.mean([q["cpu_mw"] for q in proben
                                 if q.get("phase", "").startswith("leerlauf")]))
    print(f"CPU-Leistung im Leerlauf: {leerlauf_mw:.0f} mW")

    zeilen = []
    for p in phasen:
        dauer = p["ende"] - p["start"]
        bps = p["bilder"] / dauer if p["bilder"] else None
        zeilen.append({
            "phase": p["phase"], "variante": p["variante"], "threads": p["threads"],
            "dauer_s": round(dauer, 1), "bilder": p["bilder"],
            "bilder_pro_sekunde": round(bps, 2) if bps else "",
            "proben": p["proben"], "cpu_mw": round(p["cpu_mw"], 1),
            "cpu_gpu_ane_mw": round(p["cpu_gpu_ane_mw"], 1),
            "energie_pro_bild_mj_brutto": round(p["cpu_mw"] / bps, 1) if bps else "",
            "energie_pro_bild_mj_netto": round((p["cpu_mw"] - leerlauf_mw) / bps, 1) if bps else "",
        })
        if bps:
            print(f"{p['variante']:40s} {p['threads']} Thread(s): {p['cpu_mw']:6.0f} mW, "
                  f"{bps:5.1f} Bilder/s, netto {zeilen[-1]['energie_pro_bild_mj_netto']} mJ pro Bild "
                  f"({p['proben']} Proben)")
    schreibe_csv(zeilen, "energie_phasen.csv")

    beginn = phasen[0]["start"]
    schreibe_csv([{"sekunde": round(q["ende"] - beginn, 2), "cpu_mw": q["cpu_mw"],
                   "cpu_gpu_ane_mw": q["cpu_gpu_ane_mw"], "phase": q.get("phase", "")}
                  for q in proben], "energie_proben.csv")
    print(f"Gespeichert: ergebnisse/energie_phasen.csv, energie_proben.csv, "
          f"Rohdaten daten/energie/{roh.name}")
    print("powermetrics läuft evtl. noch kurz weiter und beendet sich von selbst.")


if __name__ == "__main__":
    main()