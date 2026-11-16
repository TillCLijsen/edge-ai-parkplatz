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
| E10 | 16.11.2026 | Ausschnitt über das gedrehte Rechteck (rotatedRect) der XML, aufgerichtet mit langer Seite senkrecht, skaliert auf 64×64 Pixel, ImageNet-Normalisierung; eine gemeinsame Funktion für Training, Evaluation und Betrieb (src/vorverarbeitung.py) | 224×224 (Standard für ImageNet-Modelle); fertige Ausschnitte aus PKLotSegmented | Gemessene Platzgrösse: kurze Seite meist 32–68 px, lange Seite meist 52–121 px; 224 hiesse 4–5-fach hochskalieren bei ca. 12-facher Rechenlast; gemeinsame Funktion verhindert Abweichungen zwischen Training und Betrieb |
| E11 | 16.11.2026 | Training: ImageNet-vortrainiertes MobileNetV3-Small (Vergleich: ResNet-18), 2 Klassen, AdamW (lr 1e-3, weight decay 1e-4), Cosine-Lernrate, Batch 256, max. 15 Epochen, Abbruch nach 3 Epochen ohne Verbesserung; Augmentierung: horizontal spiegeln, 180°-Drehung, Helligkeit/Kontrast ±30 %; Auswahl der Epoche nach Balanced Accuracy auf der Validierung; Training auf Apple-GPU (MPS); 3 Seeds (42, 43, 44) | Training von Grund auf; Auswahl nach Accuracy; Training auf CPU | Vortraining spart Daten und Zeit; Balanced Accuracy ist robust gegen die je Menge schwankende Belegungsquote; Augmentierung bildet Fahrtrichtung und Wetter nach; Training ist nicht Teil der Edge-Messung; mehrere Seeds zeigen die Streuung, da MPS nicht bitgenau reproduzierbar ist |
| E12 | 16.11.2026 | Testmengen werden erst nach Festlegung aller Trainingseinstellungen und nur einmal pro Modell ausgewertet; keine Einstellung wird anhand der Testergebnisse angepasst | Hyperparameter mit Blick auf Test A/B wählen | Sonst wären die Testergebnisse optimistisch verzerrt |

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

## Herausforderungen und Fehler (für Kapitel 4a)

| Datum | Problem | Wie entdeckt | Lösung |
|---|---|---|---|
| 16.11.2026 | Ausschnitte quer liegender Plätze wurden gespiegelt (affine Abbildung mit Determinante −1) | Vergleich mit PKLotSegmented (src/pruefe_ausschnitte.py): PUCPR-Korrelation nur 0,543 im Median, Vergleichsgrafik zeigte gespiegelte Ausschnitte | Abbildung auf reine Drehung korrigiert; danach Median 0,973, Minimum 0,838, Label 296/296 identisch |

## Läufe

| Datum | Lauf | Ergebnis Validierung | Dauer |
|---|---|---|---|
| 16.11.2026 | Ausschnitte vorberechnet | Training 116.934, Validierung 28.812, Test A 125.884, Test B 205.921 Ausschnitte | 59 s gesamt |
| 16.11.2026 | mobilenet_v3_small, Seed 42 | beste Balanced Accuracy 0,99899 (Epoche 4), Abbruch nach Epoche 7 | 13,8 min (MPS) |
| 16.11.2026 | mobilenet_v3_small, Seed 43 | beste Balanced Accuracy 0,99917, Abbruch nach Epoche 13 | 27,0 min (MPS) |
| 16.11.2026 | mobilenet_v3_small, Seed 44 | beste Balanced Accuracy 0,99854 (Epoche 1), Abbruch nach Epoche 4 | 8,0 min (MPS) |
| 16.11.2026 | resnet18, Seed 42 | beste Balanced Accuracy 0,99826 (Epoche 5), Abbruch nach Epoche 8 | 29,2 min (MPS) |
| 16.11.2026 | resnet18, Seed 43 | beste Balanced Accuracy 0,99878 (Epoche 4), Abbruch nach Epoche 7 | 25,9 min (MPS) |
| 16.11.2026 | resnet18, Seed 44 | beste Balanced Accuracy 0,99759 (Epoche 1), Abbruch nach Epoche 4 | 13,9 min (MPS) |
