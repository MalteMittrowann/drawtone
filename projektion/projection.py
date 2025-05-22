import cv2
import numpy as np
import time
from screeninfo import get_monitors
from PIL import Image, ImageDraw, ImageFont

def overlay_text(image, text, position=(10, 30), font_size=32, color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_animation_tiles(image, kachelgröße):
    h, w, _ = image.shape
    tiles = []
    for i in range(0, h, kachelgröße):
        for j in range(0, w, kachelgröße):
            y1 = i
            y2 = min(i + kachelgröße, h)
            x1 = j
            x2 = min(j + kachelgröße, w)
            tiles.append((y1, y2, x1, x2))
    return tiles

def projection(image, analysebilder, analysewerte, aufbaudauer=2.0, kachelgröße=25, bottom_space=500):
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

    # Hauptbild vorbereiten
    bild_height = monitor_height - bottom_space
    bild_width = monitor_width
    bild_resized = cv2.resize(image, (bild_width, bild_height))
    anim_frame = np.zeros_like(bild_resized)
    bild_tiles = create_animation_tiles(bild_resized, kachelgröße)

    # Analysebilder vorbereiten
    analysen_resized = []
    for name, vis in analysebilder.items():
        resized = cv2.resize(vis, (int(monitor_width / len(analysebilder)), bottom_space))
        text = f"{name.replace('_', ' ').capitalize()}: {analysewerte.get(name, 0):.2f}"
        with_text = overlay_text(resized, text, position=(10, 10), font_size=24)
        analysen_resized.append(with_text)
    analysen_row = np.hstack(analysen_resized)

    anim_analysen = np.zeros_like(analysen_row)
    analysen_tiles = create_animation_tiles(analysen_row, kachelgröße)

    # Fenster einrichten
    cv2.namedWindow("Projektion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Projektion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Projektion", x, y)

    num_tiles = max(len(bild_tiles), len(analysen_tiles))
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        progress = min(elapsed / aufbaudauer, 1.0)
        tiles_to_show = int(progress * num_tiles)

        # Hauptbild aufbauen
        for i in range(tiles_to_show):
            if i < len(bild_tiles):
                y1, y2, x1, x2 = bild_tiles[i]
                anim_frame[y1:y2, x1:x2] = bild_resized[y1:y2, x1:x2]

        # Analysebilder aufbauen
        for i in range(tiles_to_show):
            if i < len(analysen_tiles):
                y1, y2, x1, x2 = analysen_tiles[i]
                anim_analysen[y1:y2, x1:x2] = analysen_row[y1:y2, x1:x2]

        # Kombinieren und anzeigen
        full_frame = np.vstack([anim_frame, anim_analysen])
        cv2.imshow("Projektion", full_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        if progress >= 1.0:
            time.sleep(0.5)  # kleine Pause am Ende
            break

    cv2.destroyAllWindows()
