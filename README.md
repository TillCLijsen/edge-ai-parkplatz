# Edge AI: Parkplatzbelegung auf dem Edge Device

Prototyp eines Edge-AI-Systems, das in Kamerabildern eines Parkplatzes jeden
Stellplatz als *frei* oder *belegt* klassifiziert. Projektbericht im IU-Kurs
„Projekt: Edge AI“ (DLBAIPEAI01_D), Aufgabenstellung 1.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Getestet mit Python 3.12.10 auf einem MacBook (Apple M2, 24 GB RAM).

## Datensatz

PKLot (Almeida et al., 2015), Lizenz CC BY 4.0.
Download: http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
Das Archiv wird nach `daten/` entpackt (nicht im Repo).

## Ausführung

Alle Schritte sind Python-Module im Ordner `src/` und werden aus dem Projektordner
gestartet (Terminal oder PyCharm-Ausführungskonfiguration mit „Module name“).
Reihenfolge:

| Nr. | Befehl | Ergebnis |
|---|---|---|
| 1 | `python -m src.datenuebersicht` | `ergebnisse/datenuebersicht_bilder.csv` |
| 2 | `python -m src.split` | `ergebnisse/split_tage.csv` (Tage → Training/Validierung/Test) |
| 3 | `python -m src.pruefe_ausschnitte` | Vergleich unserer Ausschnitte mit PKLotSegmented |
| 4 | `python -m src.ausschnitte_vorberechnen` | `daten/cache/` (64×64-Ausschnitte) |
| 5 | `python -m src.baselines` | `ergebnisse/baselines.csv` |
| 6 | `python -m src.training --modell mobilenet_v3_small --seed 42` | `modelle/*.pt`, `ergebnisse/training/` |
| 7 | `python -m src.evaluation` | `ergebnisse/evaluation_pytorch.csv` |
| 8 | `python -m src.export_onnx` | `modelle/onnx/*_fp32.onnx` mit Paritätstest |
| 9 | `python -m src.pruning` | gepruntes Modell (50 %) als `.pt` und `.onnx` |
| 10 | `python -m src.quantisierung` | `modelle/onnx/*_int8.onnx` |
| 11 | `python -m src.messung` | Grösse, Genauigkeit, Latenz, Arbeitsspeicher in `ergebnisse/` |

Gemeinsam genutzte Module: `konfiguration.py` (Pfade, Seed), `vorverarbeitung.py`
(Ausschneiden und Normalisieren, identisch in Training und Betrieb),
`datensatz.py`, `modelle.py`, `metriken.py`.

## Ordnerstruktur

| Ordner | Inhalt |
|---|---|
| `src/` | Quellcode |
| `daten/` | Datensatz und Splits (nicht im Repo) |
| `modelle/` | Trainierte Modelle (nicht im Repo) |
| `ergebnisse/` | Messwerte, Tabellen, Abbildungen |
| `docs/` | Entscheidungsprotokoll |

## Reproduzierbarkeit

Alle Zufallsgeneratoren werden über `src/konfiguration.py` mit Seed 42 gesetzt.
