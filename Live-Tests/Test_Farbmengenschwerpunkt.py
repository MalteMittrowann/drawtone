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

def berechne_farbschwerpunkt_index(image, sättigungs_schwelle=20):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))
    pixels = pixels[pixels[:, 1] > sättigungs_schwelle]  # Nur gesättigte Farben

    if len(pixels) == 0:
        return 0.0, None

    # Farben auf Einheitskreis (Hue als Winkel)
    hue = pixels[:, 0] * 2  # OpenCV Hue [0–180] → [0–360]
    hue_rad = np.deg2rad(hue)
    x = np.cos(hue_rad)
    y = np.sin(hue_rad)

    # Mittelpunkt der Hue-Verteilung (Vektoraddition)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    konzentration = np.sqrt(mean_x**2 + mean_y**2)

    schwerpunkt_index = 1.0 - konzentration

    # Für Debug: Farbrichtungen visualisieren
    debug_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    center = (150, 150)
    scale = 100
    for angle in hue_rad[::len(hue_rad)//500 + 1]:
        end = (int(center[0] + scale * np.cos(angle)), int(center[1] + scale * np.sin(angle)))
        cv2.line(debug_img, center, end, (200, 200, 200), 1)

    end_mean = (int(center[0] + scale * mean_x), int(center[1] + scale * mean_y))
    cv2.arrowedLine(debug_img, center, end_mean, (0, 0, 255), 2, tipLength=0.1)
    cv2.circle(debug_img, center, scale, (0, 0, 0), 1)
    return schwerpunkt_index, debug_img

def live_debug_schwerpunkt(kamera_index=0):
    cap = cv2.VideoCapture(kamera_index)
    if not cap.isOpened():
        print("❌ Webcam konnte nicht geöffnet werden.")
        return

    print("🎥 Drücke ESC zum Beenden.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame konnte nicht gelesen werden.")
            break

        frame_resized = cv2.resize(frame, (640, 480))
        index, debug_img = berechne_farbschwerpunkt_index(frame_resized)

        anzeige = frame_resized.copy()
        cv2.putText(anzeige, f"Farb-Schwerpunkt: {index:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 100, 255), 3, cv2.LINE_AA)

        if debug_img is not None:
            debug_img_resized = cv2.resize(debug_img, (anzeige.shape[1], anzeige.shape[0]))
            kombiniert = np.hstack((anzeige, debug_img_resized))
        else:
            kombiniert = anzeige

        cv2.imshow("Live Farb-Schwerpunkt-Analyse", kombiniert)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_debug_schwerpunkt()
