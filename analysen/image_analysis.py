import cv2
import numpy as np

# ---------------------- Helligkeit ---------------------- #
def berechne_durchschnittshelligkeit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# ---------------------- Farbanteile ---------------------- #
def berechne_farbanteile(image, thresholdWhite=75, thresholdBlack=25):
    """
    Bestimmt den Anteil von Rot, Grün, Blau, Gelb, Weiß und Schwarz im Bild.
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
        "gelb": 0,
        "weiß": 0,
        "schwarz": 0
    }

    for row in img:
        for pixel in row:
            b, g, r = pixel

            # Weiß- und Schwarz-Erkennung
            if max(r, g, b) < thresholdBlack:
                farben["schwarz"] += 1
            elif min(r, g, b) > 255 - thresholdWhite:
                farben["weiß"] += 1
            else:
                # Farberkennung: max-Kanal bestimmt Grundfarbe
                if abs(r - g) < 65 and b < min(r, g):
                    farben["gelb"] += 1
                elif g > r and g > b:
                    farben["grün"] += 1
                elif b > r and b > g:
                    farben["blau"] += 1
                elif r > g and r > b:
                    farben["rot"] += 1
                else:
                    # Wenn nichts klar zugeordnet: ignorieren
                    pass

    # Prozentuale Anteile berechnen
    farbanteile = {farbe: count / total_pixels for farbe, count in farben.items()}
    return farbanteile

# ------------------------- Segmentierungsgrad ------------------------- #
def berechne_segmentierungsgrad(image, anzahl_cluster=20, farbschwelle=25):
    # In CIELab konvertieren für bessere Farbdistanzanalyse
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab.reshape((-1, 3))

    # Chroma = Farbabstand von Grau (aus a und b)
    a = pixels[:, 1].astype(np.int16) - 128
    b = pixels[:, 2].astype(np.int16) - 128
    chroma = np.sqrt(a**2 + b**2)

    # Nur farbige Pixel berücksichtigen
    mask = chroma > farbschwelle
    relevante_pixel = pixels[mask]

    if len(relevante_pixel) < anzahl_cluster:
        return 0.0  # zu wenig relevante Pixel

    relevante_pixel = relevante_pixel.astype(np.float32)

    # k-Means-Clustering
    kriterien = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, _ = cv2.kmeans(relevante_pixel, anzahl_cluster, None, kriterien, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Clustergrößen auswerten
    _, counts = np.unique(labels, return_counts=True)

    # Segmentierungsgrad = Streuung der Clustergrößen
    segmentierungsgrad = counts.std() / counts.mean()

    return segmentierungsgrad

# ---------------------- Bildfrequenzanalyse ---------------------- #
def berechne_frequenz_index(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = np.log1p(magnitude_spectrum)  # Log-Skalierung

    mitte = magnitude_spectrum.shape[0] // 2
    radius = mitte // 4  # Kleinerer Low-Freq-Radius
    y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
    mask = (x - mitte)**2 + (y - mitte)**2 <= radius**2

    low_freq = np.sum(magnitude_spectrum[mask])
    high_freq = np.sum(magnitude_spectrum[~mask])

    if low_freq == 0:
        return 0
    return (high_freq ** 1.2) / (low_freq + 1e-6)  # Empfindlichere Gewichtung


# --------------------------- Farbharmonie --------------------------- #
def berechne_farbharmonie(image, anzahl_cluster=6, sättigungs_schwelle=20):
    # Bild in HSV konvertieren
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))

    # Nur farbige Pixel verwenden (Sättigung > Schwelle)
    pixels = pixels[pixels[:, 1] > sättigungs_schwelle]
    if len(pixels) < anzahl_cluster:
        return 0.0  # Nicht genug farbige Pixel vorhanden

    pixels = np.float32(pixels)

    # KMeans-Clustering auf farbige Pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixels, anzahl_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Cluster-Häufigkeiten
    counts = np.bincount(labels.flatten())
    total_weight = 0
    total_distance = 0

    # HS-Abstand zwischen allen Clusterpaaren berechnen (gewichtet)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            # Farbdistanz im HS-Raum (Hue zyklisch, Sat linear)
            h1, s1 = centers[i][0], centers[i][1]
            h2, s2 = centers[j][0], centers[j][1]
            dh = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 180.0  # normiert auf 0–1
            ds = abs(s1 - s2) / 255.0
            dist = np.sqrt(dh**2 + ds**2)

            # Gewicht = Produkt der beiden Clusterhäufigkeiten
            weight = counts[i] * counts[j]
            total_distance += dist * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    durchschnittliche_distanz = total_distance / total_weight

    # Harmoniewert invertieren (0 = disharmonisch, 1 = harmonisch)
    harmonie_index = 1.0 - durchschnittliche_distanz
    harmonie_index = max(0.0, min(1.0, harmonie_index))  # Begrenzung auf [0, 1]

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

    return index, varianz

# ------------------------- Farbschwerpunkt-Index -------------------------- #
def berechne_farbschwerpunkt_index(image, sättigungs_schwelle=20):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))
    pixels = pixels[pixels[:, 1] > sättigungs_schwelle]  # Nur gesättigte Farben

    if len(pixels) == 0:
        return 0.0

    # Farben auf Einheitskreis (Hue als Winkel)
    hue = pixels[:, 0] * 2  # OpenCV Hue [0–180] → [0–360]
    hue_rad = np.deg2rad(hue)
    x = np.cos(hue_rad)
    y = np.sin(hue_rad)

    # Mittelpunkt der Hue-Verteilung (Vektoraddition)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    konzentration = np.sqrt(mean_x**2 + mean_y**2)

    # Der Wert liegt zwischen 0 (völlig zerstreut) und 1 (alle Farben gleich)
    # Wir kehren ihn um, damit hoher Wert = hohe Streuung (wie bei Segmentierung)
    schwerpunkt_index = 1.0 - konzentration

    return max(0.0, min(1.0, schwerpunkt_index))
