import cv2
import numpy as np

# ------------------------- Bildrausch-Index -------------------------- #

def visualisiere_bildrausch(image, index):
    vis = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_8U)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis, f"Rausch: {index:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    return vis

# ------------------------- Farbharmonie -------------------------- #

def berechne_farbharmonie(image, anzahl_cluster=6, sättigungs_schwelle=20):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))
    pixels = pixels[pixels[:, 1] > sättigungs_schwelle]

    if len(pixels) < anzahl_cluster:
        return 0.0, np.zeros((100, 300, 3), dtype=np.uint8)

    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(pixels, anzahl_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    total_weight = 0
    total_distance = 0

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
        return 0.0, np.zeros((100, 300, 3), dtype=np.uint8)

    durchschnittliche_distanz = total_distance / total_weight
    harmonie_index = 1.0 - durchschnittliche_distanz
    harmonie_index = max(0.0, min(1.0, harmonie_index))

    # Balken-Visualisierung der Farben
    sort_idx = np.argsort(-counts)
    sorted_centers = centers[sort_idx]
    sorted_counts = counts[sort_idx]
    farbbalken = np.zeros((100, 300, 3), dtype=np.uint8)
    start_x = 0

    for i in range(len(sorted_centers)):
        color = np.uint8([[sorted_centers[i]]])
        bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0].tolist()
        breite = int(300 * sorted_counts[i] / np.sum(counts))
        cv2.rectangle(farbbalken, (start_x, 0), (start_x + breite, 100), bgr_color, -1)
        start_x += breite

    return harmonie_index, farbbalken


def visualisiere_farbharmonie(image):
    harmonie_index, farbbalken = berechne_farbharmonie(image)
    visualisierung = farbbalken.copy()

    # Text darüberlegen
    cv2.putText(
        visualisierung, f"Harmonie: {harmonie_index:.2f}",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2, cv2.LINE_AA
    )

    return visualisierung

# ------------------------- Farbschwerpunkt -------------------------- #
def berechne_farbschwerpunkt(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))
    farbschwerpunkt = np.mean(pixels, axis=0)
    return farbschwerpunkt

def visualisiere_farbschwerpunkt(image):
    h, s, v = berechne_farbschwerpunkt(image)
    farbe = np.uint8([[[h, s, v]]])
    farbe_rgb = cv2.cvtColor(farbe, cv2.COLOR_HSV2BGR)[0, 0]
    vis = np.full_like(image, farbe_rgb)
    cv2.putText(vis, f"Farbzentrum: H={int(h)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return vis

# ------------------------- Frequenzanalyse -------------------------- #
def berechne_frequenzverteilung(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    spectrum = np.log1p(magnitude)
    return spectrum

def visualisiere_frequenzanalyse(image):
    spectrum = berechne_frequenzverteilung(image)
    norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    vis = cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.putText(vis, "Frequenzanalyse", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 100), 3)
    return vis

# ------------------------- Segmentierung -------------------------- #

import cv2
import numpy as np

# ------------------- Analysefunktion ------------------- #
def berechne_segmentierungsgrad_mit_farbschwelle(image, anzahl_cluster=12, farbschwelle=25):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab.reshape((-1, 3))

    # Chroma = Farbabstand von Grau
    a = pixels[:, 1].astype(np.int16) - 128
    b = pixels[:, 2].astype(np.int16) - 128
    chroma = np.sqrt(a**2 + b**2)

    mask = chroma > farbschwelle
    relevante_pixel = pixels[mask]

    if len(relevante_pixel) < anzahl_cluster:
        leeres_bild = np.zeros_like(image)
        return 0.0, leeres_bild

    relevante_pixel = relevante_pixel.astype(np.float32)

    kriterien = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, zentren = cv2.kmeans(relevante_pixel, anzahl_cluster, None, kriterien, 10, cv2.KMEANS_RANDOM_CENTERS)

    clustered = np.zeros_like(pixels)
    clustered_pixels = zentren[labels.flatten().astype(int)]
    clustered[mask] = clustered_pixels
    clustered = clustered.reshape(lab.shape)

    clustered_bgr = cv2.cvtColor(clustered, cv2.COLOR_Lab2BGR)

    _, counts = np.unique(labels, return_counts=True)
    segmentierungsgrad = counts.std() / counts.mean()

    return segmentierungsgrad, clustered_bgr

# ------------------- Visualisierung für Projektion ------------------- #

def visualisiere_segmentierung(image):
    segmentierungsgrad, clusterbild = berechne_segmentierungsgrad_mit_farbschwelle(image)

    # Linke Seite: Originalbild mit Text
    anzeige = image.copy()
    cv2.putText(
        anzeige, f"Segmentierung: {segmentierungsgrad:.2f}", (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4, cv2.LINE_AA
    )

    # Größen ggf. angleichen
    if clusterbild.shape != anzeige.shape:
        clusterbild = cv2.resize(clusterbild, (anzeige.shape[1], anzeige.shape[0]))

    kombiniert = np.hstack((anzeige, clusterbild))
    return kombiniert


# ------------------------- Farbanteile -------------------------- #
def berechne_farbanteile(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    farben = hsv[:, :, 0].flatten()
    hist = cv2.calcHist([farben], [0], None, [12], [0, 180])
    hist_norm = hist / hist.sum()
    return hist_norm

def visualisiere_farbanteile(image):
    hist = berechne_farbanteile(image)
    breite = 50
    hoehe = 200
    bild = np.zeros((hoehe, breite * len(hist), 3), dtype=np.uint8)
    for i, h in enumerate(hist):
        farbe = np.uint8([[[i * 15, 255, 255]]])
        bgr = cv2.cvtColor(farbe, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.rectangle(bild, (i * breite, hoehe), ((i + 1) * breite, hoehe - int(h * hoehe)), bgr, -1)
    vis = cv2.resize(bild, (image.shape[1], image.shape[0]))
    cv2.putText(vis, "Farbanteile", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return vis
