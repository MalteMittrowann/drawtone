import cv2
import numpy as np
import time
from screeninfo import get_monitors

def animate_text_ticker(img, texts, y_pos, speed_px_per_sec, start_time):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    font_thickness = 10
    space_px = 300
    text_widths = [cv2.getTextSize(text, font, font_scale, font_thickness)[0][0] for text in texts]
    banner_length = sum(text_widths) + space_px * (len(texts) - 1)
    elapsed = time.time() - start_time
    x_start = int((elapsed * speed_px_per_sec) % (banner_length + img.shape[1])) - banner_length
    x = x_start
    overlay = img.copy()
    for i, text in enumerate(texts):
        cv2.putText(overlay, text, (x, y_pos), font, font_scale, (0,255,0), font_thickness)
        x += text_widths[i] + space_px
    alpha = 0.5
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def projection(image, analysis_text_lines, aufbaudauer=2.0, kachelgröße=25, bottom_space=300):
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
        print(f"📺 Zweitmonitor: Position = ({x}, {y}), Größe = {monitor_width}x{monitor_height}")

    h, w, _ = image.shape
    bild_height = monitor_height - bottom_space
    bild_width = monitor_width
    bild_resized = cv2.resize(image, (bild_width, bild_height))

    anim_frame = np.zeros_like(bild_resized)

    # Kacheln vorbereiten
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

        overlay_frame = anim_frame.copy()
        ticker_img = np.zeros_like(overlay_frame)
        ticker_img = animate_text_ticker(ticker_img, analysis_text_lines, y_pos=bild_height - 30, speed_px_per_sec=750, start_time=start_time)
        final_frame = cv2.addWeighted(ticker_img, 0.4, overlay_frame, 0.6, 0)

        cv2.imshow(fenstername, final_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
