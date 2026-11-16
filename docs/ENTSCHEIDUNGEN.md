# Entscheidungsprotokoll

Jede Entwurfs- und Implementierungsentscheidung mit Datum, Alternativen und Begründung.
Grundlage für Kapitel „Prozessreflexion und Dokumentation“ (Aufgabenteil 4a).

| Nr. | Datum | Entscheidung | Alternativen | Begründung |
|---|---|---|---|---|
| E1 | 16.11.2026 | Aufgabenstellung 1 (Bildklassifikation Smart City), Szenario Parkplatzbelegung | Aufgabe 2 (Luftqualität), Aufgabe 3 (Wake Word) | Grösster messbarer Spielraum für Optimierung (Quantisierung, Pruning, Architektur); gelabelter Datensatz vorhanden; Laptop als Zielgerät zulässig |
| E2 | 16.11.2026 | Projekt komplett neu aufgebaut, kein Code aus früheren Projekten | Wiederverwendung mit Quellenangabe | Saubere Trennung, keine Eigenplagiat-Fragen |
| E3 | 16.11.2026 | Python 3.12.10 statt 3.9.6 | Python 3.9 (vorinstalliert) | Aktuelle PyTorch- und ONNX-Runtime-Versionen setzen Python ≥ 3.11 voraus |
| E4 | 16.11.2026 | Original-PKLot (UFPR, 1280×720, XML) statt Kaggle/Roboflow-Export | Roboflow-Export „PKLot v2 640“ | Export: Bilder auf 640×640 gestreckt, achsenparallele Boxen statt gedrehter Rechtecke, keine Wetterlabels; Original ist Primärquelle |
| E5 | 16.11.2026 | Eigener Split nach Aufnahmetagen | Vorgegebener Split des Exports | Im Export liegen 99 von 100 Aufnahmetagen gleichzeitig in train, valid und test; Bilder im 5-Minuten-Takt sind fast identisch → Leakage |
| E6 | 16.11.2026 | Training PyTorch, Inferenz und Quantisierung ONNX Runtime | TensorFlow Lite, MediaPipe | Beide in der Aufgabenstellung genannt; ONNX läuft auf vielen Plattformen (Übertragbarkeit, Teil 4b) |
| E7 | 16.11.2026 | Zielgerät MacBook M2, nur CPU, 1 und 4 Threads | GPU/Neural Engine nutzen | Annäherung an schwächere Edge-Geräte; Vergleich von Hardware-Settings |

## Befunde Roboflow-Export (Grundlage für E4 und E5)

- 12.416 Bilder, 711.856 Annotationen, Klassen `space-empty` und `space-occupied`
- Parkplatz über Anzahl Plätze pro Bild bestimmbar: 28 = UFPR04 (3.517 Bilder), 40 = UFPR05 (4.152), 100 = PUCPR (4.473)
- Zeiträume ohne Überschneidung: PUCPR 11.09.–20.11.2012, UFPR04 07.12.2012–29.01.2013, UFPR05 22.02.–16.04.2013
- 274 Bilder ohne Annotation (28./29.12.2012)
- Anteil belegt: UFPR04 46,8 %, UFPR05 58,7 %, PUCPR 43,0 %
