"""Ausschneiden und Vorverarbeiten der Stellplätze.

Diese Datei wird von Training, Evaluation UND Live-Betrieb verwendet.
Nur so ist sichergestellt, dass das Modell im Betrieb exakt dieselben
Eingaben sieht wie im Training (siehe docs/ENTSCHEIDUNGEN.md, E10).
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

EINGABE_GROESSE = 64  # Pixel, quadratisch (E10)

# Mittelwert und Standardabweichung von ImageNet, weil das Modell
# mit ImageNet-Gewichten vortrainiert ist
MITTELWERT = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STANDARDABWEICHUNG = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def lese_plaetze(xml_pfad):
    """Liest alle Stellplätze aus einer PKLot-XML-Datei.

    Gibt eine Liste von Dictionaries zurück. Das Label ist 0 (frei),
    1 (belegt) oder None (kein Label vorhanden).
    """
    plaetze = []
    for platz in ET.parse(xml_pfad).getroot().iter("space"):
        rechteck = platz.find("rotatedRect")
        if rechteck is None:
            continue
        status = platz.get("occupied")
        plaetze.append({
            "id": int(platz.get("id")),
            "mitte": (float(rechteck.find("center").get("x")),
                      float(rechteck.find("center").get("y"))),
            "groesse": (float(rechteck.find("size").get("w")),
                        float(rechteck.find("size").get("h"))),
            "winkel": float(rechteck.find("angle").get("d")),
            "label": int(status) if status in ("0", "1") else None,
        })
    return plaetze


def schneide_platz_aus(bild, platz):
    """Schneidet einen Stellplatz aufgerichtet aus (lange Seite senkrecht).

    Das gedrehte Rechteck wird per affiner Abbildung in ein achsenparalleles
    Bild übertragen. Ergebnis: PIL-Bild in Originalauflösung des Platzes.
    """
    mx, my = platz["mitte"]
    breite, hoehe = platz["groesse"]
    w = math.radians(platz["winkel"])
    cos_w, sin_w = math.cos(w), math.sin(w)

    if hoehe >= breite:
        # Ausgabe-x läuft entlang der Breite, Ausgabe-y entlang der Höhe
        aus_b, aus_h = breite, hoehe
        a, b = cos_w, -sin_w
        d, e = sin_w, cos_w
    else:
        # Platz liegt quer: um 90° weiterdrehen, damit die lange Seite senkrecht ist.
        # Reine Drehung (Determinante +1), also keine Spiegelung des Ausschnitts.
        aus_b, aus_h = hoehe, breite
        a, b = -sin_w, -cos_w
        d, e = cos_w, -sin_w
    c = mx - a * aus_b / 2 - b * aus_h / 2
    f = my - d * aus_b / 2 - e * aus_h / 2

    groesse = (max(1, round(aus_b)), max(1, round(aus_h)))
    return bild.transform(groesse, Image.Transform.AFFINE, (a, b, c, d, e, f),
                          resample=Image.Resampling.BILINEAR)


def skaliere(ausschnitt):
    """Skaliert einen Ausschnitt auf 64×64 und gibt ein uint8-Array (64, 64, 3) zurück.

    In dieser Form werden die Ausschnitte für das Training zwischengespeichert.
    """
    klein = ausschnitt.resize((EINGABE_GROESSE, EINGABE_GROESSE),
                              Image.Resampling.BILINEAR)
    return np.asarray(klein, dtype=np.uint8)


def normalisiere(werte):
    """uint8-Array (..., 64, 64, 3) → float32-Array (..., 3, 64, 64).

    Werte auf 0..1 bringen, mit ImageNet-Statistik normalisieren und die
    Kanäle nach vorne stellen, wie von PyTorch und ONNX erwartet.
    """
    werte = werte.astype(np.float32) / 255.0
    werte = (werte - MITTELWERT) / STANDARDABWEICHUNG
    return np.ascontiguousarray(np.moveaxis(werte, -1, -3))


def bereite_platz_vor(bild, platz):
    """Ein Stellplatz: Ausschneiden, Skalieren und Normalisieren."""
    return normalisiere(skaliere(schneide_platz_aus(bild, platz)))


def bereite_bild_vor(bild, plaetze):
    """Alle Stellplätze eines Kamerabilds als Stapel (N, 3, 64, 64)."""
    bild = bild.convert("RGB")
    ausschnitte = np.stack([skaliere(schneide_platz_aus(bild, p)) for p in plaetze])
    return normalisiere(ausschnitte)
