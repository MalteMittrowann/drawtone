import cv2
import numpy as np

def berechne_segmentierungsgrad(image_path, k=5):
    # Bild laden
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Fehler: Bild konnte nicht geladen werden.")
        return

    # Optional: Bildgröße reduzieren (schneller)
    image = cv2.resize(image, (320, 240))

    # Pixeldaten vorbereiten
    pixels = image.reshape(-1, 3).astype(np.float32)
    print(f"[DEBUG] Pixels shape: {pixels.shape}, dtype: {pixels.dtype}")

    # KMeans Parameter
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS

    if pixels.shape[0] < k:
        print(f"⚠️ Zu wenige Pixel für {k} Cluster.")
        return

    # KMeans Clustering
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, flags)

    # Ergebnisse anzeigen
    print(f"✅ K-Means erfolgreich mit {k} Clustern.")
    print(f"Cluster-Zentren (RGB):\n{centers.astype(np.uint8)}")

    # Segmentierungsgrad berechnen (Entropie)
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
    print(f"📊 Segmentierungsgrad (Entropie): {entropy:.4f}")

    return entropy

# Beispiel: Bild analysieren
if __name__ == "__main__":
    testbild = "captured_images/tests/aufnahme_2025-05-10_11-13-13.jpg"  # <-- Pfad zu deinem Testbild
    berechne_segmentierungsgrad(testbild, k=5)