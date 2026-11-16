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
