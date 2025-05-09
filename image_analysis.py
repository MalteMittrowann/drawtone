import cv2
import numpy as np

def berechne_durchschnittshelligkeit(image):
    # In Graustufen umwandeln
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Durchschnitt berechnen
    return np.mean(gray)

def berechne_farbanteile(image):
    # Bild in RGB aufteilen (OpenCV nutzt BGR, daher umkehren)
    b, g, r = cv2.split(image)

    # Gesamtanzahl an Pixeln
    total = image.shape[0] * image.shape[1] * 255  # Maximaler Farbwert pro Kanal

    # Summen je Farbkanal (normalisiert auf 0..1 Bereich)
    rot_anteil = np.sum(r) / total
    gruen_anteil = np.sum(g) / total
    blau_anteil = np.sum(b) / total

    return {
        "Rot": round(rot_anteil, 3),
        "Grün": round(gruen_anteil, 3),
        "Blau": round(blau_anteil, 3)
    }