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
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab.reshape((-1, 3))

    a = pixels[:, 1].astype(np.int16) - 128
    b = pixels[:, 2].astype(np.int16) - 128
    chroma = np.sqrt(a**2 + b**2)

    mask = chroma > farbschwelle
    relevante_pixel = pixels[mask]

    if len(relevante_pixel) < anzahl_cluster:
        dummy_bild = np.zeros_like(image)  # oder image.copy(), wenn lieber das Original
        return 0.0, dummy_bild

    relevante_pixel = relevante_pixel.astype(np.float32)

    kriterien = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, zentren = cv2.kmeans(relevante_pixel, anzahl_cluster, None, kriterien, 10, cv2.KMEANS_RANDOM_CENTERS) # type: ignore

    clustered = np.zeros_like(pixels)
    clustered_pixels = zentren[labels.flatten().astype(int)]
    clustered[mask] = clustered_pixels
    clustered = clustered.reshape(lab.shape)

    clustered_bgr = cv2.cvtColor(clustered, cv2.COLOR_Lab2BGR)

    _, counts = np.unique(labels, return_counts=True)
    segmentierungsgrad = counts.std() / counts.mean()

    return segmentierungsgrad, clustered_bgr

# ---------------------- Bildfrequenzanalyse ---------------------- #
def berechne_frequenz_index(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = np.log1p(magnitude_spectrum)  # Log-Skalierung

    # Berechnung für Projektion
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT) # type: ignore
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    spectrum = np.log1p(magnitude)

    mitte = magnitude_spectrum.shape[0] // 2
    radius = mitte // 4
    y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
    mask = (x - mitte)**2 + (y - mitte)**2 <= radius**2

    low_freq = np.sum(magnitude_spectrum[mask])
    high_freq = np.sum(magnitude_spectrum[~mask])

    if low_freq == 0:
        dummy_spectrum = np.zeros_like(gray, dtype=np.float32)
        return 0, dummy_spectrum

    freq_index = (high_freq ** 1.2) / (low_freq + 1e-6)
    return freq_index, spectrum


# --------------------------- Farbharmonie --------------------------- #
def berechne_farbharmonie(image, anzahl_cluster=6, sättigungs_schwelle=20):
    # Bild in HSV konvertieren
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))

    # Nur farbige Pixel verwenden (Sättigung > Schwelle)
    pixels = pixels[pixels[:, 1] > sättigungs_schwelle]
    if len(pixels) < anzahl_cluster:
        dummy_balken = np.zeros((100, 300, 3), dtype=np.uint8)
        return 0.0, dummy_balken

    pixels = np.float32(pixels)

    # KMeans-Clustering auf farbige Pixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, anzahl_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) # type: ignore

    # Cluster-Häufigkeiten
    counts = np.bincount(labels.flatten())
    total_weight = 0
    total_distance = 0

    # HS-Abstand zwischen allen Clusterpaaren berechnen (gewichtet)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            h1, s1 = centers[i][0], centers[i][1]
            h2, s2 = centers[j][0], centers[j][1]
            dh = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 180.0
            ds = abs(s1 - s2) / 255.0
            dist = np.sqrt(dh**2 + ds**2)
            weight = counts[i] * counts[j]
            total_distance += dist * weight
            total_weight += weight

    if total_weight == 0:
        dummy_balken = np.zeros((100, 300, 3), dtype=np.uint8)
        return 0.0, dummy_balken

    durchschnittliche_distanz = total_distance / total_weight

    # Harmoniewert invertieren
    harmonie_index = 1.0 - durchschnittliche_distanz
    harmonie_index = max(0.0, min(1.0, harmonie_index))

    # Balken-Visualisierung der Farben
    sort_idx = np.argsort(-counts)
    sorted_centers = centers[sort_idx]
    sorted_counts = counts[sort_idx]
    farbbalken = np.zeros((100, 300, 3), dtype=np.uint8)
    start_x = 0

    for i in range(len(sorted_centers)):
        color = np.uint8([[sorted_centers[i]]]) # type: ignore
        bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0].tolist() # type: ignore
        breite = int(300 * sorted_counts[i] / np.sum(counts))
        cv2.rectangle(farbbalken, (start_x, 0), (start_x + breite, 100), bgr_color, -1)
        start_x += breite

    return harmonie_index, farbbalken

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

    # Vorläufiger Schwerpunkt auf allen Pixeln (wird ggf. bei Fehlerfall gebraucht)
    farbschwerpunkt = np.mean(pixels, axis=0)

    # Nur gesättigte Farben berücksichtigen
    pixels = pixels[pixels[:, 1] > sättigungs_schwelle]

    if len(pixels) == 0:
        visualisierung = np.ones((300, 300, 3), dtype=np.uint8) * 255
        return 0.0, farbschwerpunkt, visualisierung

    hue = pixels[:, 0] * 2
    hue_rad = np.deg2rad(hue)
    x = np.cos(hue_rad)
    y = np.sin(hue_rad)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    konzentration = np.sqrt(mean_x**2 + mean_y**2)
    schwerpunkt_index = 1.0 - konzentration
    schwerpunkt_index = max(0.0, min(1.0, schwerpunkt_index))

    visualisierung = np.ones((300, 300, 3), dtype=np.uint8) * 255
    center = (150, 150)
    scale = 100
    for angle in hue_rad[::len(hue_rad)//500 + 1]:
        end = (int(center[0] + scale * np.cos(angle)), int(center[1] + scale * np.sin(angle)))
        cv2.line(visualisierung, center, end, (200, 200, 200), 1)

    end_mean = (int(center[0] + scale * mean_x), int(center[1] + scale * mean_y))
    cv2.arrowedLine(visualisierung, center, end_mean, (0, 0, 255), 2, tipLength=0.1)
    cv2.circle(visualisierung, center, scale, (0, 0, 0), 1)

    return schwerpunkt_index, farbschwerpunkt, visualisierung


