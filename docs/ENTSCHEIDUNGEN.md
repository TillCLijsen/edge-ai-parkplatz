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
| E8 | 16.11.2026 | Split nach Tagen, geschichtet nach Regentagen: UFPR04/UFPR05 je 50 % Test A, Rest 80/20 Training/Validierung; PUCPR 50 % Test B, 50 % nur für Anpassungsversuch (src/split.py, Seed 42) | Zufallssplit über Bilder; Split über alle Parkplätze gemeinsam | Protokoll der Datensatzautoren (Tage nur in einer Menge, 50/50); Schichtung sichert Regen in jeder Testmenge; PUCPR bleibt als fremder Parkplatz vor und nach Anpassung vergleichbar |
| E9 | 16.11.2026 | Plätze ohne `occupied`-Label und das Bild ohne XML werden ausgeschlossen | Label schätzen oder aus PKLotSegmented übernehmen | Keine unbelegten Annahmen in Trainings- oder Testdaten |

## Befunde Roboflow-Export (Grundlage für E4 und E5)

- 12.416 Bilder, 711.856 Annotationen, Klassen `space-empty` und `space-occupied`
- Parkplatz über Anzahl Plätze pro Bild bestimmbar: 28 = UFPR04 (3.517 Bilder), 40 = UFPR05 (4.152), 100 = PUCPR (4.473)
- Zeiträume ohne Überschneidung: PUCPR 11.09.–20.11.2012, UFPR04 07.12.2012–29.01.2013, UFPR05 22.02.–16.04.2013
- 274 Bilder ohne Annotation (28./29.12.2012)
- Anteil belegt: UFPR04 46,8 %, UFPR05 58,7 %, PUCPR 43,0 %

## Befunde Original-PKLot (src/datenuebersicht.py, 16.11.2026)

| Parkplatz | Bilder mit XML | Tage | Plätze/Bild | gelabelte Plätze | Anteil belegt | ohne Label | Regentage | Ø JPEG |
|---|---|---|---|---|---|---|---|---|
| UFPR04 | 3.791 | 30 | 28 | 105.845 | 43,6 % | 303 | 14 | 301 KB |
| UFPR05 | 4.152 | 33 | 40 | 165.785 | 58,8 % | 295 | 8 | 259 KB |
| PUCPR | 4.473 | 36 | 100 | 422.125 | 45,5 % | 25.175 | 8 | 346 KB |

- 1 Bild ohne XML: `PUCPR/Sunny/2012-11-06/2012-11-06_18_48_46.jpg` → ausgeschlossen
- 778 Bilder mit Plätzen ohne `occupied`-Attribut; betroffene Plätze werden ausgeschlossen, nicht geraten
- Wetter ist pro Bild (Ordner) vergeben; ein Tag kann mehrere Wetterlagen haben
- Zahlen für UFPR04 und UFPR05 stimmen mit Tabelle 1 der Datensatzbeschreibung (PklotInfo.pdf) überein; PUCPR: Beschreibung nennt 424.269 Plätze, in den XML-Dateien sind 422.125 gelabelt
- Datensatzbeschreibung nennt für UFPR05 „45 parking spaces“, die XML-Dateien enthalten 40 Plätze pro Bild
- Lizenz laut Datei `licence`: CC BY 4.0
- Die Datensatzbeschreibung schreibt vor, dass Bilder desselben Tages nur in Training *oder* Test vorkommen dürfen (Abschnitt 3.1) → bestätigt E5
