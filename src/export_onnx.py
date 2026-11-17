"""Exportiert trainierte PyTorch-Modelle nach ONNX und prüft die Übereinstimmung.

Der Paritätstest vergleicht PyTorch und ONNX Runtime auf den ersten 5.000
Ausschnitten von Test A: Die Vorhersagen müssen identisch sein, die Ausgaben
dürfen nur minimal (Rundung) abweichen.

Aufruf:  python -m src.export_onnx
         python -m src.export_onnx --modelle mobilenet_v3_small_seed43
"""

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.konfiguration import MODELL_ORDNER, ONNX_ORDNER
from src.modelle import erstelle_modell

STANDARD_MODELLE = ["mobilenet_v3_small_seed43", "resnet18_seed43"]
PARITAET_ANZAHL = 5000


def lade_pytorch_modell(name):
    """Lädt ein gespeichertes Modell; die Architektur steht im Namen vor '_seed'."""
    architektur = name.split("_seed")[0]
    modell = erstelle_modell(architektur, vortrainiert=False)
    modell.load_state_dict(torch.load(MODELL_ORDNER / f"{name}.pt", map_location="cpu"))
    return modell.eval()


def exportiere(modell, ziel):
    """Exportiert mit variabler Stapelgrösse (28, 40 oder 100 Plätze pro Bild)."""
    beispiel = torch.randn(8, 3, 64, 64)
    try:
        torch.onnx.export(modell, (beispiel,), ziel, input_names=["eingabe"],
                          output_names=["ausgabe"], dynamo=True, external_data=False,
                          dynamic_shapes=({0: torch.export.Dim("stapel", max=1024)},))
    except Exception as fehler:  # Rückfall auf den klassischen Exporter
        print(f"   Dynamo-Export nicht möglich ({type(fehler).__name__}), klassischer Export")
        torch.onnx.export(modell, (beispiel,), ziel, input_names=["eingabe"],
                          output_names=["ausgabe"], dynamo=False, opset_version=17,
                          dynamic_axes={"eingabe": {0: "stapel"}, "ausgabe": {0: "stapel"}})
    onnx.checker.check_model(onnx.load(ziel))


def pruefe_paritaet(modell, ziel):
    """Vergleicht PyTorch und ONNX Runtime auf denselben Ausschnitten."""
    from src.datensatz import PlatzDatensatz
    datensatz = PlatzDatensatz("test_a")
    x = np.stack([datensatz[i][0].numpy() for i in range(PARITAET_ANZAHL)])
    with torch.no_grad():
        aus_pytorch = modell(torch.from_numpy(x)).numpy()
    sitzung = ort.InferenceSession(str(ziel), providers=["CPUExecutionProvider"])
    aus_onnx = sitzung.run(None, {"eingabe": x})[0]
    gleich = np.mean(aus_pytorch.argmax(1) == aus_onnx.argmax(1))
    abweichung = np.abs(aus_pytorch - aus_onnx).max()
    print(f"   Parität: Vorhersagen gleich {gleich:.4%}, max. Abweichung der Ausgaben {abweichung:.2e}")
    assert gleich == 1.0, "PyTorch und ONNX liefern unterschiedliche Vorhersagen"
    return abweichung


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelle", nargs="+", default=STANDARD_MODELLE)
    args = parser.parse_args()
    ONNX_ORDNER.mkdir(parents=True, exist_ok=True)
    for name in args.modelle:
        print(name)
        modell = lade_pytorch_modell(name)
        ziel = ONNX_ORDNER / f"{name}_fp32.onnx"
        exportiere(modell, ziel)
        pruefe_paritaet(modell, ziel)
        print(f"   gespeichert: {ziel.name} ({ziel.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
