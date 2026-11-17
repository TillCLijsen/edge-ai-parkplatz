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
| E13 | 17.11.2026 | Hauptmodell MobileNetV3-Small Seed 43, Vergleich ResNet-18 Seed 43 (je beste Validierung); Zusatzversuch „Inbetriebnahme am neuen Standort“ gestrichen | Auswahl nach Testergebnis; Anpassungsversuch trotzdem durchführen | Auswahl nur über Validierung (E12); auf Test B gibt es keinen Genauigkeitsverlust, der sich durch Nachtraining aufholen liesse, ein nachträglich erschwerter Transfer wäre gezielte Effektsuche |
| E14 | 17.11.2026 | Optimierungsvarianten: ONNX FP32, INT8 statisch (QDQ, Gewichte pro Kanal, Kalibrierung mit 1.000 Trainingsausschnitten, MinMax), 50 % globales L1-Pruning mit 2 Epochen Nachtraining, Pruning + INT8; Referenz: PyTorch FP32 auf CPU | Dynamische Quantisierung; strukturiertes Pruning; FP16 | Statische Quantisierung quantisiert auch die Aktivierungen und eignet sich für CNNs; unstrukturiertes Pruning ist der einfache Standard, sein fehlender Latenzgewinn auf dichter CPU-Hardware ist selbst ein Ergebnis; FP16 weggelassen, weil die Aufgabenstellung Quantisierung, Pruning und leichte Architekturen nennt und das Seitenlimit wenige, gut begründete Varianten verlangt |
| E15 | 17.11.2026 | Messaufbau: nur CPU, 1 und 4 Threads, jede Messung in eigenem Prozess, 10 Bilder Aufwärmen, 3 Runden über 30 Testbilder je Parkplatz, Median und 95. Perzentil, 20 s Pause zwischen Läufen; gemessen wird die ganze Kette (JPEG dekodieren, ausschneiden, Inferenz, JSON) | Nur Inferenzzeit; Mittelwert | Die Anforderung ≤ 1 s gilt für die ganze Verarbeitung eines Kamerabilds; Median und P95 sind robust gegen Ausreisser; Pausen, weil das MacBook Air ohne Lüfter unter Dauerlast drosseln kann |
| E16 | 17.11.2026 | INT8-Quantisierung mit Mischpräzision: die ersten zwei Faltungen und die Knoten dazwischen bleiben in FP32, alles andere INT8 (gleiche Regel für alle Modelle) | Alles INT8 mit MinMax (ursprünglich E14); Percentile-Kalibrierung; Quantisierung mit 16-Bit-Aktivierungen; quantisierungsbewusstes Training | Analyse auf der Validierung (src/quantisierung_analyse.py): nur die Ausgabe der ersten Faltung ist empfindlich (breiter Wertebereich, die meisten Kanäle nutzen nur einen kleinen Teil davon); Percentile verbessert, reicht aber nicht; 16-Bit-Aktivierungen sind auf der CPU nicht schneller als FP32; quantisierungsbewusstes Training wäre deutlich aufwendiger als nötig |
| E17 | 17.11.2026 | Energiemessung mit powermetrics (Sampler cpu_power, alle 500 ms): 30 s Leerlauf vorher und nachher, je Variante und Thread-Anzahl 30 s Dauerbetrieb mit PUCPR-Testbildern (ganze Kette wie in der Latenzmessung), 20 s Pause; Proben nur, wenn vollständig in einer Phase (1 s Rand verworfen); Energie pro Bild brutto (Leistung unter Last / Bilder pro Sekunde) und netto (abzüglich Leerlauf); Varianten: PyTorch-Referenz, MobileNet FP32 und INT8, ResNet-18 FP32 und INT8 | Externes Strommessgerät an der Steckdose; Messung über den Akkustand | Kein Messgerät vorhanden; powermetrics ist auf macOS eingebaut und trennt CPU von Bildschirm und Peripherie; der Akkustand ist zu grob; Einschränkung: powermetrics liefert eine Schätzung aus Apples Energiemodell, keine elektrische Messung |
| E18 | 17.11.2026 | Simulierter Live-Betrieb: pro Parkplatz ein zufälliger Testtag (Seed 42) in Aufnahmereihenfolge, MobileNet INT8 mit 1 Thread, JSON-Meldungen in eine Datei statt an einen Server; zusätzlich Zählgenauigkeit (gezählte gegen echte freie Plätze pro Bild) über alle Testbilder; Vergleich nur über annotierte Plätze | Echte Kamera; Tag von Hand wählen; nur Genauigkeit pro Platz | Keine Kamera am Parkplatz verfügbar, PKLot liefert echte Bildfolgen im 5-Minuten-Takt; zufällige Tagwahl verhindert Rosinenpicken; für den Betreiber zählt die Zahl freier Plätze; 1 Thread, weil er laut Energiemessung pro Bild am wenigsten Energie braucht |
| E19 | 17.11.2026 | Code-Abgabe über den Link zum GitHub-Repo im Bericht; Repo bleibt bis kurz vor der Abgabe privat und wird dann öffentlich; vorher Prüfung auf persönliche Daten, Aufgabenstellung, Datensatz und Modelle | Code-Auszüge im Anhang; Repo dauerhaft privat mit Einladung | So bisher von den Prüfenden gewünscht; spät öffentlich, weil die Aufgabenstellung davon abrät, ausgearbeitete Lösungen zu teilen |

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
| 17.11.2026 | Messung ungültig: Skripte liefen gleichzeitig und der Mac lief auf Akku | Zeitstempel der Dateien überlappten, resnet18_int8 fehlte in der Messung, messaufbau.json zeigte netzteil_angeschlossen: false | Alle Läufe gestoppt, Kette nacheinander mit Netzteil neu gestartet; Skripte schreiben ihre Ausgaben seither zusätzlich nach ergebnisse/protokolle/ |
| 17.11.2026 | Pruning machte das Modell unbrauchbar (Balanced Accuracy 0,5, nur noch eine Klasse) | Messung der Genauigkeit nach dem Pruning, danach Anteil Nullen je Schicht in der ONNX-Datei geprüft: Ausgabeschicht zu 100 % genullt, Nachtraining kann genullte Gewichte nicht zurückholen | Ausgabeschicht vom Pruning ausgenommen; Skript meldet jetzt Schichten mit mehr als 90 % Nullen |
| 17.11.2026 | INT8 bei MobileNetV3-Small verlor stark an Genauigkeit (Test A 0,70, Test B 0,80; Validierung 0,72) | Messung der Genauigkeit; Ursache auf der Validierung gesucht: Gewichte allein auf 8 Bit unkritisch, dann jeden Zwischenwert einzeln auf 8 Bit gerundet: die Ausgabe der ersten Faltung ändert allein 31 % der Vorhersagen | Mischpräzision (E16); Auswahl nur über die Validierung, die Testergebnisse des ersten Laufs bleiben als gescheiterter Versuch dokumentiert |
| 17.11.2026 | Energiemessung: powermetrics schrieb keine Daten (Rohdatei blieb leer) | Erste Vermutung (im Hintergrund gestarteter Prozess wird beendet) war falsch; die Fehlermeldung der Shell zeigte „Operation not permitted“ beim Schreiben in den Projektordner | Ursache: macOS schützt den Ordner Schreibtisch auch vor dem mit Administratorrechten gestarteten Prozess; powermetrics schreibt jetzt nach /tmp, das Skript kopiert die Rohdaten danach nach daten/energie/ |

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
| 17.11.2026 | ONNX-Export MobileNet und ResNet (Seed 43) | Parität zu PyTorch auf 5.000 Ausschnitten aus Test A: 100 % gleiche Vorhersagen, max. Abweichung 8,8e-05 bzw. 1,6e-05 | ca. 1 min |
| 17.11.2026 | Pruning 50 %, erster Versuch (Ausgabeschicht mitgeprunt) | 0,50000 direkt danach und nach beiden Epochen, verworfen | ca. 2 min |
| 17.11.2026 | Pruning 50 %, Ausgabeschicht ausgenommen | 0,99880 direkt danach, 0,99878 nach Epoche 1 (gewählt), 0,99853 nach Epoche 2; vollständig genullt: SE-Schichten in features.1 (100 %), fast vollständig: features.4 (96–97 %) | ca. 2,5 min |
| 17.11.2026 | Quantisierungsanalyse MobileNet | FP32 0,99917; INT8 MinMax 0,72000; INT8 Percentile 0,96274; INT8 mit ersten zwei Faltungen in FP32 0,99892; empfindlichster Zwischenwert: Ausgabe der ersten Faltung (Bereich −154,9 bis 164,6, ändert 29,8 % der Vorhersagen), alle anderen ≤ 0,2 % | ca. 3 min |
| 17.11.2026 | Messung (Netzteil, nacheinander) | siehe Abschnitt „Ergebnisse Messung“ | ca. 37 min |
| 17.11.2026 | Energiemessung (Netzteil, src/energiemessung.py) | siehe Abschnitt „Ergebnisse Energie“; je Lastphase 53–55 Proben à 500 ms | ca. 12 min |
| 17.11.2026 | Live-Betrieb (src/live_betrieb.py) | siehe Abschnitt „Ergebnisse Live-Betrieb“ | ca. 1 min |
| 17.11.2026 | Fehleranalyse (src/fehleranalyse.py) | siehe Abschnitt „Ergebnisse Fehleranalyse“ | ca. 1 min |

## Ergebnisse Messung (17.11.2026, ergebnisse/messung_*.csv)

MacBook Air M2 (Mac14,2), macOS 26.6.2, ONNX Runtime 1.32.0, Netzteil angeschlossen. Latenz = ganze Kette pro Kamerabild, Median über 90 Bilder.

| Variante | Grösse (gzip) | Bal. Acc. Test A / Test B | Fehler pro 10.000 Plätze A / B | PUCPR 1 Thread | PUCPR 4 Threads | RAM-Spitze |
|---|---|---|---|---|---|---|
| PyTorch FP32 (Referenz) | – | wie ONNX FP32 | – | 577,9 ms | 510,5 ms | 584,6 MB |
| MobileNet ONNX FP32 | 6,37 MB (5,66) | 0,9988 / 0,9992 | 11,76 / 7,24 | 128,9 ms | 47,1 ms | 202,4 MB |
| MobileNet INT8 | 1,80 MB (1,49) | 0,9987 / 0,9990 | 13,35 / 9,47 | 30,6 ms | 24,4 ms | 193,4 MB |
| MobileNet gepruned FP32 | 6,37 MB (3,41) | 0,9987 / 0,9993 | 13,27 / 6,75 | 128,8 ms | 47,0 ms | 202,3 MB |
| MobileNet gepruned INT8 | 1,80 MB (1,11) | 0,9985 / 0,9993 | 14,62 / 7,43 | 30,3 ms | 24,3 ms | 189,2 MB |
| ResNet-18 ONNX FP32 | 44,78 MB (41,54) | 0,9982 / 0,9984 | 18,27 / 14,96 | 558,1 ms | 222,3 ms | 255,3 MB |
| ResNet-18 INT8 | 11,48 MB (9,66) | 0,9982 / 0,9985 | 18,11 / 14,37 | 146,7 ms | 76,8 ms | 211,0 MB |

Befunde:
- Alle Varianten erfüllen ≤ 1 s pro Bild deutlich, auch PyTorch FP32.
- INT8 bei MobileNet: Verlust Balanced Accuracy 0,01 (Test A) bzw. 0,02 Prozentpunkte (Test B), Konfidenzintervalle überlappen; Inferenz bei 1 Thread 114,2 → 15,4 ms (7,4-fach), ganze Kette 4,2-fach schneller; gegenüber PyTorch 18,9-fach.
- Bei INT8 und 4 Threads ist die Vorverarbeitung (JPEG dekodieren und ausschneiden, 15,3 ms) teurer als die Inferenz (9,1 ms): der Engpass liegt danach nicht mehr im Modell.
- Pruning: keine Änderung bei Latenz und unkomprimierter Grösse (Nullen werden weiter gespeichert und gerechnet), nur komprimiert kleiner (5,66 → 3,41 MB); Genauigkeit unverändert.
- RAM-Spitze wird von Python und ONNX Runtime bestimmt (Grundlast ca. 51 MB, Spitze 190–200 MB für alle MobileNet-Varianten), kaum vom Modell.
- Übertragung: Meldung als JSON 135–351 Byte statt JPEG 258–333 KB pro Bild (ca. 1.000-fach weniger Daten).
- ResNet-18 INT8: Mit der einheitlichen Regel (E16) bleiben auch dort die ersten zwei Faltungen in FP32; dadurch langsamer als im Lauf vom 17.11.2026, 11:47 Uhr ohne diese Regel (gleicher Messaufbau, dort 105,2 ms bei 1 Thread und 61,4 ms bei 4 Threads, Genauigkeit gleich). Wird im Bericht als Folge der einheitlichen Regel genannt.

## Ergebnisse Energie (17.11.2026, ergebnisse/energie_phasen.csv)

powermetrics, Sampler cpu_power, 500 ms; 30 s Dauerbetrieb pro Phase mit PUCPR-Testbildern (100 Plätze); Netzteil angeschlossen. CPU-Leistung im Leerlauf: 1.150 mW vorher, 1.397 mW nachher, Mittel 1.273 mW.

| Variante | Threads | Bilder/s | CPU-Leistung | Energie pro Bild brutto | Energie pro Bild netto |
|---|---|---|---|---|---|
| PyTorch FP32 | 1 | 1,49 | 5.787 mW | 3.877 mJ | 3.024 mJ |
| PyTorch FP32 | 4 | 1,89 | 12.347 mW | 6.524 mJ | 5.851 mJ |
| MobileNet ONNX FP32 | 1 | 6,68 | 5.216 mW | 781 mJ | 590 mJ |
| MobileNet ONNX FP32 | 4 | 19,93 | 16.380 mW | 822 mJ | 758 mJ |
| MobileNet INT8 | 1 | 28,75 | 6.256 mW | 218 mJ | 173 mJ |
| MobileNet INT8 | 4 | 37,78 | 17.226 mW | 456 mJ | 422 mJ |
| ResNet-18 ONNX FP32 | 1 | 1,56 | 5.680 mW | 3.632 mJ | 2.817 mJ |
| ResNet-18 ONNX FP32 | 4 | 4,09 | 20.016 mW | 4.893 mJ | 4.582 mJ |
| ResNet-18 INT8 | 1 | 5,93 | 6.905 mW | 1.165 mJ | 950 mJ |
| ResNet-18 INT8 | 4 | 12,65 | 18.725 mW | 1.481 mJ | 1.380 mJ |

Befunde:
- MobileNet INT8 mit 1 Thread braucht netto 173 mJ pro Bild: 3,4-mal weniger als ONNX FP32 (590 mJ) und 17,5-mal weniger als die PyTorch-Referenz (3.024 mJ).
- 4 Threads sind schneller, kosten aber bei allen Varianten mehr Energie pro Bild (MobileNet INT8: 422 statt 173 mJ). Für den Dauerbetrieb ist 1 Thread sparsamer, 4 Threads lohnen sich nur, wenn die Latenz zählt.
- CPU+GPU+ANE (combined_power) ist praktisch gleich der CPU-Leistung: GPU und Neural Engine werden nicht genutzt, die Messung ist tatsächlich reine CPU-Last.
- Unsicherheit durch den Leerlauf: Grundlast stieg von 1.150 auf 1.397 mW. Mit dem einen oder anderen Wert statt des Mittels ändert sich die Netto-Energie bei MobileNet INT8 (1 Thread) um etwa ±2,5 %; die Reihenfolge der Varianten bleibt gleich.
- Im Dauerbetrieb etwas langsamer als in der Latenzmessung (z. B. PyTorch 1 Thread 1,49 statt 1,73 Bilder/s), vermutlich Drosselung des lüfterlosen MacBook Air unter Dauerlast; nicht separat gemessen.
- Einschränkung: powermetrics schätzt die Leistung aus Apples Energiemodell; keine elektrische Messung, Bildschirm und Peripherie nicht enthalten.

## Ergebnisse Live-Betrieb (17.11.2026, ergebnisse/live_zaehlung.csv)

MobileNet INT8, 1 Thread; Zählfehler = gezählte minus echte freie Plätze pro Bild, nur annotierte Plätze.

| Quelle | Parkplatz | Bilder | mittl. abs. Fehler | exakt | höchstens 1 daneben | max. Fehler | Latenz Median / max |
|---|---|---|---|---|---|---|---|
| Live-Tag 2012-12-12 (Mi) | UFPR04 | 120 | 0,000 | 100 % | 100 % | 0 | 11,7 / 22,2 ms |
| Live-Tag 2013-03-19 (Di) | UFPR05 | 142 | 0,056 | 94,4 % | 100 % | 1 | 15,7 / 61,7 ms |
| Live-Tag 2012-10-28 (So) | PUCPR | 156 | 0,000 | 100 % | 100 % | 0 | 30,3 / 71,8 ms |
| alle Testbilder | UFPR04 | 1.840 | 0,038 | 96,4 % | 99,8 % | 3 | – |
| alle Testbilder | UFPR05 | 1.869 | 0,050 | 95,6 % | 99,5 % | 3 | – |
| alle Testbilder | PUCPR | 2.160 | 0,083 | 93,3 % | 99,1 % | 12 | – |

Befunde:
- Echtzeit: Bildabstand im Datensatz 5 min, längste Verarbeitung 71,8 ms; Reserve über drei Grössenordnungen.
- Die zufällig gezogenen Live-Tage sind wenig repräsentativ: PUCPR war ein Sonntag mit 98–100 freien Plätzen, UFPR04 fast voll (0–5 frei). Die 100 % dort sagen wenig; massgeblich für die Zählgenauigkeit sind die Zahlen über alle Testbilder. Tag wurde nachträglich nicht geändert (sonst Rosinenpicken), im Bericht offen benennen.
- Mittlerer Fehler (Vorzeichen) nahe 0 (0,000 bis 0,023): keine systematische Über- oder Unterzählung.
- Ausreisser: ein PUCPR-Bild mit 12 Plätzen Abweichung; wird in der Fehleranalyse untersucht.

## Ergebnisse Fehleranalyse (17.11.2026, ergebnisse/fehler_*.csv, abbildungen/)

MobileNet INT8 auf Test A und B: 331.805 Ausschnitte, 156 frei als belegt erkannt, 207 belegt als frei erkannt (zusammen 363, passt zu messung_genauigkeit.csv).

Fehler nach Wetter (pro 10.000 Plätze der jeweiligen Klasse):

| Menge | Wetter | frei als belegt (pro 10.000 freie) | belegt als frei (pro 10.000 belegte) | Fehler absolut |
|---|---|---|---|---|
| Test A | Cloudy | 8,75 | 11,11 | 15 + 27 |
| Test A | Rainy | 3,90 | 47,92 | 2 + 13 |
| Test A | Sunny | 13,27 | 15,87 | 54 + 57 |
| Test B | Cloudy | 2,41 | 2,26 | 15 + 5 |
| Test B | Rainy | 3,59 | 13,27 | 4 + 34 |
| Test B | Sunny | 12,23 | 22,98 | 66 + 71 |

Genauigkeit gegen Latenz (Fehler pro 10.000 Plätze Test A+B; Latenz PUCPR, 1 Thread): PyTorch FP32 8,95 / 577,9 ms; MobileNet FP32 8,95 / 128,9 ms; MobileNet INT8 10,94 / 30,6 ms; gepruned FP32 9,22 / 128,8 ms; gepruned INT8 10,16 / 30,3 ms; ResNet-18 FP32 16,21 / 558,1 ms; ResNet-18 INT8 15,79 / 146,7 ms.

Befunde:
- Bild mit den meisten Fehlern (PUCPR, 2012-10-25 13:03, Regen, 12 Plätze „belegt, als frei erkannt“, fehler_bild_1.png): bei Sichtprüfung sind diese 12 Plätze leer; die Annotation passt hier nicht zum Bild, das Modell liegt richtig.
- Zweitschlimmstes Bild (PUCPR, 2012-10-23 09:00, Regen, 6 Fehler, fehler_bild_2.png): in den falsch erkannten Plätzen stehen überwiegend dunkle Autos auf nassem Asphalt; echte Modellfehler (Sichtprüfung in verkleinerter Ansicht, in voller Grösse nachprüfen).
- Die 8 sichersten Fehler je Richtung (fehler_beispiele.png) sind zum Teil Grenzfälle oder fragwürdige Annotationen: leere Plätze als belegt annotiert; als frei annotierte Plätze mit durchfahrendem Auto oder Fussgängern; Nachbarauto ragt in den Platz.
- Folgerung: Ein Teil der gemessenen Fehler geht auf die Annotation zurück; die Fehlerquoten sind eher eine Obergrenze. Nur durch Sichtprüfung einzelner Beispiele gestützt, keine systematische Nachannotation; im Bericht entsprechend vorsichtig formulieren.
- Regen: belegte Plätze werden häufiger als frei erkannt (Test A 47,9 pro 10.000 belegte gegenüber 11,1 bewölkt und 15,9 sonnig; Test B 13,3 gegenüber 2,3 und 23,0). In Test A beruht der Wert auf nur 13 Fehlern; in Test B ist Sonne schlechter als Regen. Keine eindeutige Wetterabhängigkeit belegbar; zwei der drei schlimmsten Bilder stammen aber von Regentagen.
- Auffällig: PUCPR 2012-09-15 (sonnig) hat mehrere aufeinanderfolgende Bilder mit Fehlern (11:27–11:32 und 15:37–15:47), also dauerhafte Fehler an denselben Plätzen über mehrere Aufnahmen; Ursache nicht untersucht (möglicherweise Schatten).
- INT8 kostet rund 2 Fehler pro 10.000 Plätze (10,94 statt 8,95) bei 4,2-fach kürzerer Latenz; ResNet-18 ist langsamer und zugleich ungenauer als MobileNet.

