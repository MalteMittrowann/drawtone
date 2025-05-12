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
def berechne_segmentierungsgrad(image, anzahl_cluster=6):
    # Bild in Float32 und in 2D umwandeln (für kmeans)
    pixel = image.reshape((-1, 3)).astype(np.float32)

    # Kriterien für das Beenden des k-means Algorithmus
    kriterien = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    # k-Means-Clustering
    _, labels, _ = cv2.kmeans(pixel, anzahl_cluster, None, kriterien, 10, cv2.KMEANS_RANDOM_CENTERS) # type: ignore

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


# --------------------------- Farbharmonie --------------------------- #
def berechne_farbharmonie(image, anzahl_cluster=6):
    # Bild in HSV konvertieren
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Nur Farbton (H) und Sättigung (S) verwenden, um Farbabstand zu messen
    pixels = hsv.reshape((-1, 3))
    pixels = np.float32(pixels)

    # KMeans auf HSV anwenden (optional gewichtet auf H und S)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixels, anzahl_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # type: ignore

    # Nur die H-Komponente (Farbton) extrahieren
    hue_values = centers[:, 0]  # H-Komponente aus HSV

    # Zyklische Distanz im Farbkreis berechnen
    def hue_distance(h1, h2):
        d = abs(h1 - h2)
        return min(d, 180 - d)  # HSV-H geht von 0–180 in OpenCV

    # Alle Paarabstände berechnen
    total_distance = 0
    count = 0
    for i in range(len(hue_values)):
        for j in range(i + 1, len(hue_values)):
            total_distance += hue_distance(hue_values[i], hue_values[j])
            count += 1

    # Durchschnittlicher Farbtonabstand
    if count == 0:
        return 0
    durchschnittlicher_abstand = total_distance / count

    # Optional normieren (0–90) und invertieren, damit hohe Werte = harmonisch
    harmonie_index = 1 - (durchschnittlicher_abstand / 90.0)  # 0 = max Kontrast, 1 = perfekte Harmonie
    harmonie_index = max(0.0, min(1.0, harmonie_index))  # Begrenzen auf 0–1

    return harmonie_index

# ------------------------- Bildrausch-Index -------------------------- #
def berechne_bildrausch_index(image):
    # In Graustufen konvertieren
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Laplace-Operator anwenden
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Varianz der Laplace-Antwort als Maß für Bildrauschen/Unruhe
    varianz = np.var(laplacian)

    # Optional normieren (je nach Erfahrungswerten)
    # Hier: 0 = keine Kanten, >1000 = sehr detailreich
    # Du kannst den Bereich anpassen basierend auf Tests
    index = np.tanh(varianz / 500)  # Sanfte Normierung auf ca. 0–1

    return index