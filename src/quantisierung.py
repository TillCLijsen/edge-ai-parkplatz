"""Statische INT8-Quantisierung der ONNX-Modelle mit ONNX Runtime.

Gewichte und Aktivierungen werden von 32-Bit-Gleitkomma auf 8-Bit-Ganzzahlen
abgebildet. Die Wertebereiche der Aktivierungen werden vorab mit 1.000 zufällig
gezogenen Ausschnitten aus dem TRAINING kalibriert (nie aus dem Test).

Einstellungen: QDQ-Format, Gewichte pro Kanal (per_channel) als INT8,
Aktivierungen als UINT8, Kalibrierung MinMax (Standard von ONNX Runtime).
Die ersten zwei Faltungen und die Knoten dazwischen bleiben in FP32, weil die
Aktivierungen dort für 8 Bit zu ungleich verteilt sind
(Analyse: src/quantisierung_analyse.py).

Aufruf:  python -m src.quantisierung
"""

import argparse

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.ausschnitte_vorberechnen import CACHE_ORDNER
from src.konfiguration import ONNX_ORDNER, SEED, starte_protokoll
from src.vorverarbeitung import normalisiere

STANDARD_MODELLE = ["mobilenet_v3_small_seed43", "mobilenet_v3_small_seed43_pruned50",
                    "resnet18_seed43"]
KALIBRIER_ANZAHL = 1000
STAPEL = 50


class KalibrierDaten(CalibrationDataReader):
    """Liefert die Kalibrierausschnitte stapelweise an ONNX Runtime."""

    def __init__(self):
        bilder = np.load(CACHE_ORDNER / "training_bilder.npy", mmap_mode="r")
        zufall = np.random.default_rng(SEED)
        auswahl = np.sort(zufall.choice(len(bilder), KALIBRIER_ANZAHL, replace=False))
        daten = normalisiere(np.array(bilder[auswahl]))
        self.stapel = iter([{"eingabe": daten[i:i + STAPEL]} for i in range(0, len(daten), STAPEL)])

    def get_next(self):
        return next(self.stapel, None)


def bereite_vor(quelle, ziel):
    """Formen ableiten, Graph vereinfachen und jedem Knoten einen Namen geben."""
    quant_pre_process(str(quelle), str(ziel))
    modell = onnx.load(str(ziel))
    for i, knoten in enumerate(modell.graph.node):
        if not knoten.name:
            knoten.name = f"knoten_{i}_{knoten.op_type}"
    onnx.save(modell, str(ziel))
    return modell


def knoten_bis_zweite_faltung(modell):
    """Namen aller Knoten vom Eingang bis und mit der zweiten Faltung."""
    namen, faltungen = [], 0
    for knoten in modell.graph.node:
        namen.append(knoten.name)
        if knoten.op_type == "Conv":
            faltungen += 1
            if faltungen == 2:
                break
    return namen


def quantisiere(name, erste_fp32=True, methode=CalibrationMethod.MinMax, ziel=None):
    """Erzeugt die INT8-Variante eines FP32-Modells."""
    quelle = ONNX_ORDNER / f"{name}_fp32.onnx"
    vorbereitet = ONNX_ORDNER / f"{name}_fp32_vorbereitet.onnx"
    ziel = ziel or ONNX_ORDNER / f"{name}_int8.onnx"
    modell = bereite_vor(quelle, vorbereitet)
    ausnahmen = knoten_bis_zweite_faltung(modell) if erste_fp32 else []
    quantize_static(str(vorbereitet), str(ziel), KalibrierDaten(),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=methode, nodes_to_exclude=ausnahmen)
    vorbereitet.unlink()
    print(f"{name}: {quelle.stat().st_size / 1e6:.2f} MB → {ziel.stat().st_size / 1e6:.2f} MB, "
          f"in FP32 belassen: {len(ausnahmen)} Knoten")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelle", nargs="+", default=STANDARD_MODELLE)
    args = parser.parse_args()
    starte_protokoll("quantisierung")
    for name in args.modelle:
        quantisiere(name)


if __name__ == "__main__":
    main()