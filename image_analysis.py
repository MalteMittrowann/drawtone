import cv2
import numpy as np

# ---------------------- Helligkeit ----------------------
def berechne_durchschnittshelligkeit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# ---------------------- Farbanteile ----------------------
def berechne_farbanteile(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    farben = {
        "rot": ([0, 50, 50], [10, 255, 255]),
        "orange": ([11, 50, 50], [25, 255, 255]),
        "gelb": ([26, 50, 50], [35, 255, 255]),
        "gruen": ([36, 50, 50], [85, 255, 255]),
        "blau": ([86, 50, 50], [125, 255, 255]),
        "lila": ([126, 50, 50], [160, 255, 255]),
        "pink": ([161, 50, 50], [179, 255, 255]),
    }
    pixel_gesamt = image.shape[0] * image.shape[1]
    anteile = {}
    for name, (lower, upper) in farben.items():
        maske = cv2.inRange(hsv, np.array(lower), np.array(upper))
        anteile[name] = np.sum(maske > 0) / pixel_gesamt
    return anteile

# ---------------------- Segmentierungsgrad ----------------------
def berechne_segmentierungsgrad(image, anzahl_cluster=5):
    # Bild verkleinern für Performance
    img_small = cv2.resize(image, (100, 100))  # ggf. anpassen für Performance
    data = img_small.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(
        data, anzahl_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    segmented = labels.reshape((img_small.shape[:2]))
    unique, counts = np.unique(segmented, return_counts=True)
    segmentierungsgrad = len(unique)
    return segmentierungsgrad


# ---------------------- Bildfrequenzanalyse ----------------------
def berechne_frequenz_index(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Frequenzindex = Verhältnis hoher zu niedriger Frequenzen
    mitte = magnitude_spectrum.shape[0] // 2
    radius = mitte // 2
    y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
    mask = (x - mitte)**2 + (y - mitte)**2 <= radius**2

    low_freq = np.sum(magnitude_spectrum[mask])
    high_freq = np.sum(magnitude_spectrum[~mask])
    
    if low_freq == 0:
        return 0
    return high_freq / low_freq
