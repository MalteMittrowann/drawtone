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

def berechne_frequenz_index(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)

    mitte = magnitude_spectrum.shape[0] // 2
    radius = mitte // 4  # Empfindlichere Trennung
    y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
    mask = (x - mitte)**2 + (y - mitte)**2 <= radius**2

    low_freq = np.sum(magnitude_spectrum_log[mask])
    high_freq = np.sum(magnitude_spectrum_log[~mask])

    if low_freq == 0:
        index = 0
    else:
        index = (high_freq ** 1.2) / (low_freq + 1e-6)

    # Visualisierung vorbereiten
    visual = np.uint8(255 * magnitude_spectrum_log / np.max(magnitude_spectrum_log))
    visual = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)

    # Maske auf das Visualbild zeichnen
    mask_img = visual.copy()
    cv2.circle(mask_img, (mitte, mitte), radius, (0, 255, 0), 2)  # Kreis für Low-Frequency-Bereich

    return index, mask_img

# ------------------ LIVE-VORSCHAU ------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    index, freq_img = berechne_frequenz_index(frame)

    # Anzeige des Frequenzindexes im Bild
    vis = frame.copy()
    cv2.putText(vis, f"Frequenz-Index: {index:.2f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Beide Bilder nebeneinander
    freq_img_resized = cv2.resize(freq_img, (vis.shape[1], vis.shape[0]))
    combined = np.hstack((vis, freq_img_resized))

    cv2.imshow("Live + Frequenz-Analyse - Press 'q' to quit", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
