import cv2
import numpy as np

# ------------------------- Bildrausch-Index -------------------------- #
def berechne_bildrausch_index(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    varianz = np.var(laplacian)
    index = np.tanh(varianz / 500)
    return index, varianz

def visualisiere_bildrausch(image):
    index, varianz = berechne_bildrausch_index(image)
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
        return 0.0
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
            dist = np.sqrt(dh ** 2 + ds ** 2)
            weight = counts[i] * counts[j]
            total_distance += dist * weight
            total_weight += weight
    if total_weight == 0:
        return 0.0
    durchschnittliche_distanz = total_distance / total_weight
    harmonie_index = 1.0 - durchschnittliche_distanz
    return max(0.0, min(1.0, harmonie_index))

def visualisiere_farbharmonie(image):
    harmonie = berechne_farbharmonie(image)
    vis = image.copy()
    cv2.putText(vis, f"Harmonie: {harmonie:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 100, 255), 3)
    return vis

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
def berechne_segmentierungsgrad(image, cluster=12, helligkeits_thresh=25):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 2] > helligkeits_thresh
    pixel = image[mask].reshape((-1, 3)).astype(np.float32)
    if len(pixel) < cluster:
        return 0.0
    kriterien = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, _ = cv2.kmeans(pixel, cluster, None, kriterien, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    segmentierungsgrad = counts.std() / counts.mean()
    return segmentierungsgrad

def visualisiere_segmentierung(image):
    index = berechne_segmentierungsgrad(image)
    vis = image.copy()
    cv2.putText(vis, f"Segmentierung: {index:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)
    return vis

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
