"""Pruning: entfernt 50 % der Gewichte (die betragsmässig kleinsten) und trainiert kurz nach.

Verfahren: globales, unstrukturiertes L1-Pruning über alle Faltungs- und
vollverbundenen Schichten (torch.nn.utils.prune), danach 2 Epochen Nachtraining
mit kleiner Lernrate, wobei die entfernten Gewichte auf null bleiben.
Ausgewählt wird die Epoche mit der besten Balanced Accuracy auf der Validierung.

Aufruf:  python -m src.pruning
Ausgabe: modelle/mobilenet_v3_small_seed43_pruned50.pt und die ONNX-Datei dazu
"""

import argparse

import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datensatz import PlatzDatensatz
from src.export_onnx import exportiere, lade_pytorch_modell, pruefe_paritaet
from src.konfiguration import MODELL_ORDNER, ONNX_ORDNER, setze_seeds
from src.training import bewerte, waehle_geraet


def zu_prunende_gewichte(modell):
    """Alle Gewichte von Faltungs- und Linearschichten."""
    return [(m, "weight") for m in modell.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def anteil_null(modell):
    """Anteil der Gewichte, die exakt null sind (über die geprunten Schichten)."""
    gewichte = [m.weight for m, _ in zu_prunende_gewichte(modell)]
    gesamt = sum(g.numel() for g in gewichte)
    return sum((g == 0).sum().item() for g in gewichte) / gesamt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modell", default="mobilenet_v3_small_seed43")
    parser.add_argument("--anteil", type=float, default=0.5)
    parser.add_argument("--epochen", type=int, default=2)
    parser.add_argument("--lernrate", type=float, default=1e-4)
    args = parser.parse_args()

    setze_seeds(43)
    geraet = waehle_geraet()
    modell = lade_pytorch_modell(args.modell)
    prozent = round(args.anteil * 100)
    name = f"{args.modell}_pruned{prozent}"

    prune.global_unstructured(zu_prunende_gewichte(modell),
                              pruning_method=prune.L1Unstructured, amount=args.anteil)
    modell.to(geraet)
    validierung = DataLoader(PlatzDatensatz("validierung"), batch_size=512)
    _, bal_acc = bewerte(modell, validierung, geraet)
    print(f"Direkt nach dem Pruning: Val. Balanced Accuracy {bal_acc:.5f}, "
          f"Anteil Null {anteil_null(modell):.1%}")

    training = DataLoader(PlatzDatensatz("training", augmentieren=True, seed=43), batch_size=256,
                          shuffle=True, generator=torch.Generator().manual_seed(43))
    optimierer = torch.optim.AdamW(modell.parameters(), lr=args.lernrate, weight_decay=1e-4)
    verlustfunktion = nn.CrossEntropyLoss()
    beste = -1.0
    for epoche in range(1, args.epochen + 1):
        modell.train()
        for x, y in tqdm(training, desc=f"Nachtraining {epoche}", leave=False):
            x, y = x.to(geraet), y.to(geraet)
            optimierer.zero_grad()
            verlustfunktion(modell(x), y).backward()
            optimierer.step()
        _, bal_acc = bewerte(modell, validierung, geraet)
        print(f"Epoche {epoche}: Val. Balanced Accuracy {bal_acc:.5f}")
        if bal_acc > beste:
            beste = bal_acc
            # Masken fest einrechnen, damit ein normales Modell gespeichert wird
            kopie = lade_pytorch_modell(args.modell)
            zustand = {k.replace("_orig", ""): v for k, v in modell.state_dict().items()
                       if not k.endswith("_mask")}
            masken = {k.replace("_mask", ""): v for k, v in modell.state_dict().items()
                      if k.endswith("_mask")}
            for k, maske in masken.items():
                zustand[k] = zustand[k] * maske
            kopie.load_state_dict({k: v.cpu() for k, v in zustand.items()})
            torch.save(kopie.state_dict(), MODELL_ORDNER / f"{name}.pt")

    endmodell = lade_pytorch_modell(name)
    print(f"Gespeichert: {name}.pt, Anteil Null {anteil_null(endmodell):.1%}, "
          f"beste Val. Balanced Accuracy {beste:.5f}")
    ziel = ONNX_ORDNER / f"{name}_fp32.onnx"
    exportiere(endmodell, ziel)
    pruefe_paritaet(endmodell, ziel)
    print(f"ONNX: {ziel.name} ({ziel.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
