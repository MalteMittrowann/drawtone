#-----------------------------------------------------------
 # DRAWTONE
 # Copyright (c) 2026 Dave Kronawitter & Malte Mittrowann.
 # All rights reserved.
 #
 # This code is proprietary and not open source.
 # Unauthorized copying of this file is strictly prohibited.
#------------------------------------------------------------


import cv2
import numpy as np
import time
from screeninfo import get_monitors

# Hilfsfunktionen für einzelne Analysen
from projektion.analysenFuerProjektion import (
    visualisiere_bildrausch,
    visualisiere_farbanteile,
    visualisiere_farbschwerpunkt,
    visualisiere_frequenzanalyse
)

def projection(image, image_analyse, analysewerte, aufbaudauer=2.0, kachelgröße=25, bottom_space=500):
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

    h, w, _ = image.shape
    bild_height = monitor_height - bottom_space
    bild_width = monitor_width
    bild_resized = cv2.resize(image, (bild_width, bild_height))

    anim_frame = np.zeros_like(bild_resized)

    # Analysevorschauen vorbereiten (jetzt ohne Neu-Berechnung)
    vorschauen = [
        (visualisiere_bildrausch(image_analyse.copy()), analysewerte.get("bildrausch_index", None), "Bildrausch"),
        (analysewerte.get("farbbalken", None), analysewerte.get("farbharmonie", None), "Farbharmonie"),
        (visualisiere_farbanteile(image_analyse.copy()), None, "Farbanteile"),
        (visualisiere_frequenzanalyse(analysewerte.get("frequenz_spektrum", None)), analysewerte.get("frequenzverteilung", None), "Frequenz"),
        (analysewerte.get("clusterbildSegmentierungsGrad", None), analysewerte.get("segmentierungsgrad", None), "Segmentierung"),
        (visualisiere_farbschwerpunkt(image_analyse.copy(), analysewerte.get("farbschwerpunkt_projektion_farbe", None)), analysewerte.get("farbschwerpunkt", None), "Farbschwerpunkt-Farbe"),
        (analysewerte.get("farbschwerpunkt_visualisierung_pfeil", None), analysewerte.get("farbschwerpunkt", None), "Farbschwerpunkt-Farbe")
    ]

    vorschau_height = bottom_space
    einzel_width = int(monitor_width / len(vorschauen))
    analyse_frames = []
    for vorschau, wert, titel in vorschauen:
        resized = cv2.resize(vorschau, (einzel_width, vorschau_height))
        if wert is not None:
            cv2.putText(resized, f"{titel}: {wert:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(resized, titel, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        analyse_frames.append(resized)

    analyse_tiles = []
    analyse_anim_frames = []
    for analyse_img in analyse_frames:
        anim = np.zeros_like(analyse_img)
        tiles = []
        for i in range(0, analyse_img.shape[0], kachelgröße):
            for j in range(0, analyse_img.shape[1], kachelgröße):
                y1 = i
                y2 = min(i + kachelgröße, analyse_img.shape[0])
                x1 = j
                x2 = min(j + kachelgröße, analyse_img.shape[1])
                tiles.append((y1, y2, x1, x2))
        analyse_tiles.append(tiles)
        analyse_anim_frames.append(anim)

    tiles = []
    for i in range(0, bild_height, kachelgröße):
        for j in range(0, bild_width, kachelgröße):
            y1 = i
            y2 = min(i + kachelgröße, bild_height)
            x1 = j
            x2 = min(j + kachelgröße, bild_width)
            tiles.append((y1, y2, x1, x2))

    tile_count = len(tiles)
    analyse_tile_counts = [len(t) for t in analyse_tiles]

    start_time = time.time()
    last_shown_index = 0
    analyse_last_indices = [0 for _ in analyse_tiles]

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

        # Analyse-Fenster gleichzeitig animieren
        for ai, (tileset, target_img, anim_img) in enumerate(zip(analyse_tiles, analyse_frames, analyse_anim_frames)):
            expected_ai_tiles = int((elapsed / aufbaudauer) * analyse_tile_counts[ai])
            expected_ai_tiles = min(expected_ai_tiles, analyse_tile_counts[ai])
            for idx in range(analyse_last_indices[ai], expected_ai_tiles):
                y1, y2, x1, x2 = tileset[idx]
                anim_img[y1:y2, x1:x2] = target_img[y1:y2, x1:x2]
            analyse_last_indices[ai] = expected_ai_tiles

        analysen_row = np.hstack(analyse_anim_frames)
        if analysen_row.shape[1] != anim_frame.shape[1]:
            analysen_row = cv2.resize(analysen_row, (anim_frame.shape[1], analysen_row.shape[0]))
        full_frame = np.vstack([anim_frame, analysen_row])

        cv2.imshow(fenstername, full_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# Beispielaufruf (nur falls direkt ausgeführt)
if __name__ == "__main__":
    import os
    testbild = cv2.imread("testbild.jpg")
    dummy_analyse = {
        "bildrausch_index": 0.45,
        "farbharmonie": 0.82,
        "frequenzverteilung": 0.68,
        "segmentierungsgrad": 5
    }
    if testbild is not None:
        projection(testbild, testbild, dummy_analyse)
    else:
        print("❌ Bild konnte nicht geladen werden.")
