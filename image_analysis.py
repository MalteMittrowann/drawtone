import cv2
import numpy as np

# ---------------------- Helligkeit ---------------------- #
def berechne_durchschnittshelligkeit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# ---------------------- Farbanteile ---------------------- #
def berechne_farbanteile(image, threshold=100):
    """
    Bestimmt den Anteil von Grundfarben (RGB + CMY), Weiß und Schwarz im Bild.
    Threshold: Empfindlichkeit für Schwarz und Weiß (0–255).
    """
    h, w, _ = image.shape
    total_pixels = h * w

    # Bild zu float konvertieren
    img = image.astype(np.float32)

    # Zähler für Farben
    farben = {
        "rot": 0,
        "grün": 0,
        "blau": 0,
        "cyan": 0,
        "magenta": 0,
        "gelb": 0,
        "weiß": 0,
        "schwarz": 0
    }

    for row in img:
        for pixel in row:
            b, g, r = pixel

            if max(r, g, b) < threshold:
                farben["schwarz"] += 1
            elif min(r, g, b) > 255 - threshold:
                farben["weiß"] += 1
            else:
                if r > g and r > b:
                    if g > b:
                        farben["gelb"] += 1
                    elif b > g:
                        farben["magenta"] += 1
                    else:
                        farben["rot"] += 1
                elif g > r and g > b:
                    if r > b:
                        farben["gelb"] += 1
                    elif b > r:
                        farben["cyan"] += 1
                    else:
                        farben["grün"] += 1
                elif b > r and b > g:
                    if r > g:
                        farben["magenta"] += 1
                    elif g > r:
                        farben["cyan"] += 1
                    else:
                        farben["blau"] += 1

    # Prozentuale Anteile
    farbanteile = {farbe: count / total_pixels for farbe, count in farben.items()}
    return farbanteile

# ---------------------- Segmentierungsgrad ---------------------- #
def berechne_segmentierungsgrad(image, anzahl_cluster=5):
    # Bild in Float32 und in 2D umwandeln (für kmeans)
    pixel = image.reshape((-1, 3)).astype(np.float32)

    # Kriterien für das Beenden des k-means Algorithmus
    kriterien = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    # k-Means-Clustering
    _, labels, _ = cv2.kmeans(pixel, anzahl_cluster, kriterien, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Anzahl Pixel pro Cluster zählen
    _, counts = np.unique(labels, return_counts=True)

    # Einfacher Maßwert für Segmentierung: Verhältnis kleinster zu größter Cluster
    segmentierungsgrad = counts.std() / counts.mean()  # Variabilität der Clustergrößen

    return segmentierungsgrad


# ---------------------- Bildfrequenzanalyse ---------------------- #
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
