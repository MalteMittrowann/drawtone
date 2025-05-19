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
    pixels = hsv.reshape((-1, 3)).astype(np.float32)

    # KMeans auf HSV anwenden
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixels, anzahl_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # type: ignore

    hue_values = centers[:, 0]  # Nur die H-Komponente (Farbton)

    # ---------------------- Punkt 2: Farbkreis-Abstände ---------------------- #
    def hue_distance(h1, h2):
        d = abs(h1 - h2)
        return min(d, 180 - d)  # HSV-H geht von 0–180 in OpenCV

    abstaende = []
    for i in range(len(hue_values)):
        for j in range(i + 1, len(hue_values)):
            abstaende.append(hue_distance(hue_values[i], hue_values[j]))

    if not abstaende:
        return 0

    durchschnittlicher_abstand = np.mean(abstaende)

    # ---------------------- Punkt 3: Adobe-inspirierte Harmonie ---------------------- #
    def bewertung_farbmuster(hue_values):
        scores = []
        hue_values = sorted(hue_values)

        # 1. Komplementärfarben: Abstand nahe 90° (180° auf OpenCV-Skala)
        for i in range(len(hue_values)):
            for j in range(i + 1, len(hue_values)):
                d = hue_distance(hue_values[i], hue_values[j])
                diff = abs(d - 90)
                score = max(0, 1 - (diff / 90))  # perfekte Komplementärfarbe = 1
                scores.append(score)

        # 2. Triade (je 60° Abstand im 180°-Kreis)
        for i in range(len(hue_values)):
            for j in range(i + 1, len(hue_values)):
                d = hue_distance(hue_values[i], hue_values[j])
                diff = abs(d - 60)
                score = max(0, 1 - (diff / 60))
                scores.append(score)

        # 3. Analogfarben (nahe beieinander, max 20°)
        for i in range(len(hue_values)):
            for j in range(i + 1, len(hue_values)):
                d = hue_distance(hue_values[i], hue_values[j])
                score = max(0, 1 - (d / 20)) if d <= 20 else 0
                scores.append(score) 

        if not scores:
            return 0

        return np.mean(scores)

    muster_score = bewertung_farbmuster(hue_values)

    # ---------------------- Kombination & Normierung ---------------------- #
    # Durchschnittlicher Abstand normiert: 0 (chaotisch) bis 1 (eng zusammen)
    abstand_score = 1 - (durchschnittlicher_abstand / 90.0)
    abstand_score = np.clip(abstand_score, 0, 1)

    # Kombinierter Score (gewichtbar)
    harmonie_index = (abstand_score + muster_score) / 2.0
    harmonie_index = np.clip(harmonie_index, 0, 1)

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