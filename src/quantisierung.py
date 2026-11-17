"""Statische INT8-Quantisierung der ONNX-Modelle mit ONNX Runtime.

Gewichte und Aktivierungen werden von 32-Bit-Gleitkomma auf 8-Bit-Ganzzahlen
abgebildet. Die Wertebereiche der Aktivierungen werden vorab mit 1.000 zufällig
gezogenen Ausschnitten aus dem TRAINING kalibriert (nie aus dem Test).

Einstellungen: QDQ-Format, Gewichte pro Kanal (per_channel) als INT8,
Aktivierungen als UINT8, Kalibrierung MinMax (Standard von ONNX Runtime).

Aufruf:  python -m src.quantisierung
"""

import argparse

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.ausschnitte_vorberechnen import CACHE_ORDNER
from src.konfiguration import ONNX_ORDNER, SEED
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


def quantisiere(name):
    quelle = ONNX_ORDNER / f"{name}_fp32.onnx"
    vorbereitet = ONNX_ORDNER / f"{name}_fp32_vorbereitet.onnx"
    ziel = ONNX_ORDNER / f"{name}_int8.onnx"
    # Empfohlene Vorbereitung: Formen ableiten und Graph vereinfachen
    quant_pre_process(str(quelle), str(vorbereitet))
    quantize_static(str(vorbereitet), str(ziel), KalibrierDaten(),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    vorbereitet.unlink()
    print(f"{name}: {quelle.stat().st_size / 1e6:.2f} MB → {ziel.stat().st_size / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelle", nargs="+", default=STANDARD_MODELLE)
    args = parser.parse_args()
    for name in args.modelle:
        quantisiere(name)


if __name__ == "__main__":
    main()
