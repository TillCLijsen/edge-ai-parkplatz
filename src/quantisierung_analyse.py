"""Analyse der INT8-Quantisierung von MobileNetV3-Small (nur Validierungsdaten).

Teil 1 (Empfindlichkeit): Jeder Zwischenwert des FP32-Modells wird einzeln auf
8 Bit gerundet (Wertebereich aus denselben 1.000 Kalibrierausschnitten wie bei
MinMax). Gemessen wird, wie viele Vorhersagen sich auf 3.000 zufälligen
Validierungsausschnitten gegenüber FP32 ändern.
Teil 2 (Varianten): Balanced Accuracy auf der ganzen Validierung für FP32,
INT8 MinMax, INT8 Percentile und INT8 mit den ersten zwei Faltungen in FP32.

Aufruf:  python -m src.quantisierung_analyse
Ausgabe: ergebnisse/quantisierung_empfindlichkeit.csv, ergebnisse/quantisierung_varianten.csv
"""

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import helper, numpy_helper
from onnxruntime.quantization import CalibrationMethod

from src.ausschnitte_vorberechnen import CACHE_ORDNER
from src.konfiguration import ERGEBNIS_ORDNER, MODELL_ORDNER, ONNX_ORDNER, SEED, starte_protokoll
from src.quantisierung import KalibrierDaten, bereite_vor, quantisiere
from src.vorverarbeitung import normalisiere

MODELL = "mobilenet_v3_small_seed43"
ANALYSE_ORDNER = MODELL_ORDNER / "analyse"


def vorhersagen(pfad, bilder, stapel=500):
    """Vorhergesagte Klasse für jeden Ausschnitt."""
    sitzung = ort.InferenceSession(str(pfad), providers=["CPUExecutionProvider"])
    ergebnis = []
    for i in range(0, len(bilder), stapel):
        x = normalisiere(np.array(bilder[i:i + stapel]))
        ergebnis.append(sitzung.run(None, {"eingabe": x})[0].argmax(1))
    return np.concatenate(ergebnis)


def balanced_accuracy(vorhersage, label):
    return 0.5 * ((vorhersage[label == 1] == 1).mean() + (vorhersage[label == 0] == 0).mean())


def wertebereiche(modell, tensoren):
    """Minimum und Maximum jedes Zwischenwerts über die Kalibrierausschnitte."""
    infos = {v.name: v for v in onnx.shape_inference.infer_shapes(modell).graph.value_info}
    kopie = onnx.ModelProto()
    kopie.CopyFrom(modell)
    for t in tensoren:
        kopie.graph.output.append(infos[t])
    pfad = ANALYSE_ORDNER / "alle_ausgaben.onnx"
    onnx.save(kopie, str(pfad))
    sitzung = ort.InferenceSession(str(pfad), providers=["CPUExecutionProvider"])
    namen = [o.name for o in sitzung.get_outputs()]
    # Null liegt immer im Bereich (wie bei ONNX Runtime)
    unten = {t: 0.0 for t in tensoren}
    oben = {t: 0.0 for t in tensoren}
    daten = KalibrierDaten()
    while (stapel := daten.get_next()) is not None:
        werte = dict(zip(namen, sitzung.run(None, stapel)))
        for t in tensoren:
            unten[t] = min(unten[t], float(werte[t].min()))
            oben[t] = max(oben[t], float(werte[t].max()))
    return unten, oben


def mit_einem_qdq(modell, tensor, unten, oben, ziel):
    """Kopie des FP32-Modells, in der nur ein Zwischenwert auf 8 Bit gerundet wird."""
    skala = max(oben - unten, 1e-12) / 255
    nullpunkt = int(np.clip(round(-unten / skala), 0, 255))
    kopie = onnx.ModelProto()
    kopie.CopyFrom(modell)
    neue_knoten = []
    for knoten in kopie.graph.node:
        for i, eingang in enumerate(knoten.input):
            if eingang == tensor:
                knoten.input[i] = tensor + "_dq"
        neue_knoten.append(knoten)
        if tensor in knoten.output:
            # Runden auf 8 Bit direkt nach dem Knoten, der den Wert erzeugt
            neue_knoten.append(helper.make_node(
                "QuantizeLinear", [tensor, tensor + "_skala", tensor + "_null"], [tensor + "_q"]))
            neue_knoten.append(helper.make_node(
                "DequantizeLinear", [tensor + "_q", tensor + "_skala", tensor + "_null"],
                [tensor + "_dq"]))
    del kopie.graph.node[:]
    kopie.graph.node.extend(neue_knoten)
    kopie.graph.initializer.extend([
        numpy_helper.from_array(np.array(skala, np.float32), tensor + "_skala"),
        numpy_helper.from_array(np.array(nullpunkt, np.uint8), tensor + "_null")])
    onnx.save(kopie, str(ziel))


def main():
    starte_protokoll("quantisierung_analyse")
    ANALYSE_ORDNER.mkdir(parents=True, exist_ok=True)
    fp32 = ANALYSE_ORDNER / f"{MODELL}_fp32_vorbereitet.onnx"
    modell = bereite_vor(ONNX_ORDNER / f"{MODELL}_fp32.onnx", fp32)
    infos = {v.name for v in onnx.shape_inference.infer_shapes(modell).graph.value_info}
    tensoren = [a for k in modell.graph.node for a in k.output if a in infos]

    bilder = np.load(CACHE_ORDNER / "validierung_bilder.npy", mmap_mode="r")
    label = pd.read_csv(CACHE_ORDNER / "validierung_info.csv")["label"].to_numpy()

    # Teil 1: Empfindlichkeit einzelner Zwischenwerte
    auswahl = np.sort(np.random.default_rng(SEED).choice(len(bilder), 3000, replace=False))
    teil = np.array(bilder[auswahl])
    referenz = vorhersagen(fp32, teil)
    unten, oben = wertebereiche(modell, tensoren)
    zeilen = []
    for t in tensoren:
        pfad = ANALYSE_ORDNER / "ein_tensor.onnx"
        mit_einem_qdq(modell, t, unten[t], oben[t], pfad)
        geaendert = (vorhersagen(pfad, teil) != referenz).mean()
        zeilen.append({"tensor": t, "unten": round(unten[t], 3), "oben": round(oben[t], 3),
                       "geaenderte_vorhersagen_prozent": round(100 * geaendert, 2)})
    tabelle = pd.DataFrame(zeilen).sort_values("geaenderte_vorhersagen_prozent", ascending=False)
    tabelle.to_csv(ERGEBNIS_ORDNER / "quantisierung_empfindlichkeit.csv", index=False)
    print(f"{len(tensoren)} Zwischenwerte geprüft, die fünf empfindlichsten:")
    print(tabelle.head(5).to_string(index=False))

    # Teil 2: Varianten auf der ganzen Validierung
    varianten = {
        "fp32": None,
        "int8_minmax": dict(erste_fp32=False, methode=CalibrationMethod.MinMax),
        "int8_percentile": dict(erste_fp32=False, methode=CalibrationMethod.Percentile),
        "int8_minmax_erste_zwei_faltungen_fp32": dict(erste_fp32=True,
                                                      methode=CalibrationMethod.MinMax),
    }
    ergebnis = []
    for variante, einstellung in varianten.items():
        if einstellung is None:
            pfad = ONNX_ORDNER / f"{MODELL}_fp32.onnx"
        else:
            pfad = ANALYSE_ORDNER / f"{variante}.onnx"
            quantisiere(MODELL, ziel=pfad, **einstellung)
        bal_acc = balanced_accuracy(vorhersagen(pfad, bilder), label)
        ergebnis.append({"variante": variante, "val_balanced_accuracy": round(bal_acc, 5)})
        print(f"{variante:40s} Val. Balanced Accuracy {bal_acc:.5f}")
    pd.DataFrame(ergebnis).to_csv(ERGEBNIS_ORDNER / "quantisierung_varianten.csv", index=False)


if __name__ == "__main__":
    main()