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
        cv2.putText(overlay, text, (x, y_pos), font, font_scale, (255,255,255), font_thickness)
        x += text_widths[i] + space_px
    alpha = 0.5
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def projection(image, analysis_text_lines, aufbaudauer=2.0, kachelgröße=25, bottom_space=300, fade_duration=0.3):
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
    num_rows = (bild_height + kachelgröße - 1) // kachelgröße
    num_cols = (bild_width + kachelgröße - 1) // kachelgröße
    total_tiles = num_rows * num_cols

    start_time = time.time()
    tile_times = [None] * total_tiles
    shown_tiles = set()

    fenstername = "Projektion"
    cv2.namedWindow(fenstername, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(fenstername, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(fenstername, x, y)

    while True:
        elapsed = time.time() - start_time
        tiles_to_show = min(int((elapsed / aufbaudauer) * total_tiles), total_tiles)

        for tile_idx in range(tiles_to_show):
            if tile_idx not in shown_tiles:
                tile_times[tile_idx] = time.time()
                shown_tiles.add(tile_idx)

        frame = anim_frame.copy()

        for tile_idx in shown_tiles:
            i = tile_idx // num_cols
            j = tile_idx % num_cols
            y1 = i * kachelgröße
            y2 = min((i + 1) * kachelgröße, bild_height)
            x1 = j * kachelgröße
            x2 = min((j + 1) * kachelgröße, bild_width)
            t0 = tile_times[tile_idx]
            alpha = min((time.time() - t0) / fade_duration, 1.0)
            tile_src = bild_resized[y1:y2, x1:x2].astype(np.float32)
            tile_dst = frame[y1:y2, x1:x2].astype(np.float32)
            tile_mix = cv2.addWeighted(tile_dst, 1 - alpha, tile_src, alpha, 0)
            frame[y1:y2, x1:x2] = tile_mix.astype(np.uint8)

        ticker_img = np.zeros_like(frame)
        ticker_img = animate_text_ticker(ticker_img, analysis_text_lines, y_pos=bild_height - 30, speed_px_per_sec=750, start_time=start_time)
        final_frame = cv2.addWeighted(ticker_img, 0.4, frame, 0.6, 0)

        cv2.imshow(fenstername, final_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
