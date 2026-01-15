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

def nothing(x):
    pass

def circularity_live_vorschau(kamera_index=0):
    cap = cv2.VideoCapture(kamera_index)
    if not cap.isOpened():
        print("❌ Webcam konnte nicht geöffnet werden.")
        return

    # Fenster & Trackbar für Schwellenwert
    cv2.namedWindow("Circularity Live")
    cv2.createTrackbar("Schwelle x100", "Circularity Live", 70, 100, nothing)

    print("🎥 Drücke ESC zum Beenden.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame konnte nicht gelesen werden.")
            break

        # Circularity-Schwellenwert auslesen (0.00–1.00)
        schwelle = cv2.getTrackbarPos("Schwelle x100", "Circularity Live") / 100.0

        # Bildvorverarbeitung
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # Konturen finden
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anzeige = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if perimeter == 0 or area < 100:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            x, y, w, h = cv2.boundingRect(cnt)

            farbe = (0, 255, 0) if circularity >= schwelle else (0, 0, 255)  # grün oder rot

            cv2.drawContours(anzeige, [cnt], -1, farbe, 2)
            cv2.putText(anzeige, f"Circ: {circularity:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, farbe, 2)

        # Anzeige
        cv2.imshow("Circularity Live", anzeige)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Start
if __name__ == "__main__":
    circularity_live_vorschau()
