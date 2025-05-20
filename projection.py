import cv2
import numpy as np
import time
from screeninfo import get_monitors

# Hilfsfunktionen für einzelne Analysen
from analysenFuerProjektion import (
    berechne_bildrausch_index, visualisiere_bildrausch,
    berechne_farbharmonie, visualisiere_farbharmonie,
    berechne_farbschwerpunkt, visualisiere_farbschwerpunkt,
    berechne_frequenzverteilung, visualisiere_frequenzanalyse,
    berechne_segmentierungsgrad, visualisiere_segmentierung,
    berechne_farbanteile, visualisiere_farbanteile
)

def erstelle_analysen_vorschau(bild):
    vorschauen = []

    # Bildrausch
    index, _ = berechne_bildrausch_index(bild)
    v1 = visualisiere_bildrausch(bild.copy())
    vorschauen.append(v1)

    # Farbharmonie
    h = berechne_farbharmonie(bild.copy())
    v2 = visualisiere_farbharmonie(bild.copy())
    vorschauen.append(v2)

    # Farbschwerpunkt
    s = berechne_farbschwerpunkt(bild.copy())
    v3 = visualisiere_farbschwerpunkt(bild.copy())
    vorschauen.append(v3)

    # Frequenzanalyse
    f_index = berechne_frequenzverteilung(bild.copy())
    v4 = visualisiere_frequenzanalyse(bild.copy())
    vorschauen.append(v4)

    # Segmentierung
    seg = berechne_segmentierungsgrad(bild.copy(), 12, 25)
    v5 = visualisiere_segmentierung(bild.copy())
    vorschauen.append(v5)

    # Farbanteile
    v6 = visualisiere_farbanteile(bild.copy())
    vorschauen.append(v6)

    return vorschauen

def projection(image, aufbaudauer=2.0, kachelgröße=25, bottom_space=500):
    monitore = get_monitors()
    if len(monitore) < 2:
        print("⚠️ Nur ein Bildschirm erkannt.")
        x, y = 0, 0
        monitor_width = 1280
        monitor_height = 720
    else:
        monitor = monitore[1]
        x, y = monitor.x, monitor.y
        monitor_width = monitor.width
        monitor_height = monitor.height

    # Aufteilung: obere Hälfte = Bild mit Animation, untere Hälfte = Vorschauen
    h, w, _ = image.shape
    bild_height = int(monitor_height * 0.55)
    bild_width = monitor_width
    bild_resized = cv2.resize(image, (bild_width, bild_height))

    # Analysevorschauen vorbereiten (feste Höhe)
    analysen = erstelle_analysen_vorschau(image)
    vorschau_height = monitor_height - bild_height
    einzel_width = int(monitor_width / len(analysen))
    analysen_resized = [cv2.resize(a, (einzel_width, vorschau_height)) for a in analysen]
    analysen_row = np.hstack(analysen_resized)

    anim_frame = np.zeros_like(bild_resized)

    tiles = []
    for i in range(0, bild_height, kachelgröße):
        for j in range(0, bild_width, kachelgröße):
            y1 = i
            y2 = min(i + kachelgröße, bild_height)
            x1 = j
            x2 = min(j + kachelgröße, bild_width)
            tiles.append((y1, y2, x1, x2))

    tile_count = len(tiles)
    last_shown_index = 0
    start_time = time.time()

    fenstername = "Projektion"
    cv2.namedWindow(fenstername, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(fenstername, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(fenstername, x, y)

    while True:
        elapsed = time.time() - start_time
        expected_tiles = int((elapsed / aufbaudauer) * tile_count)
        expected_tiles = min(expected_tiles, tile_count)

        for idx in range(last_shown_index, expected_tiles):
            y1, y2, x1, x2 = tiles[idx]
            anim_frame[y1:y2, x1:x2] = bild_resized[y1:y2, x1:x2]
        last_shown_index = expected_tiles

        # Bild mit Vorschauen kombinieren
        full_frame = np.vstack([anim_frame, analysen_row])

        cv2.imshow(fenstername, full_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# Beispielaufruf (nur falls direkt ausgeführt)
if __name__ == "__main__":
    testbild = cv2.imread("testbild.jpg")
    if testbild is not None:
        projection(testbild)
    else:
        print("❌ Bild konnte nicht geladen werden.")
