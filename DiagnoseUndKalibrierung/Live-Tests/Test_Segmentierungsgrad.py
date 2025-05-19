import cv2
import numpy as np

# ---------------------- Analysefunktion ---------------------- #
def berechne_segmentierungsgrad_mit_farbschwelle(image, anzahl_cluster=6, farbschwelle=10):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab.reshape((-1, 3))

    # Chroma = Farbabstand von Grau (basierend auf a & b)
    a = pixels[:, 1].astype(np.int16) - 128
    b = pixels[:, 2].astype(np.int16) - 128
    chroma = np.sqrt(a**2 + b**2)

    mask = chroma > farbschwelle
    relevante_pixel = pixels[mask]

    if len(relevante_pixel) < anzahl_cluster:
        return 0.0, np.zeros_like(image)

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

# ---------------------- Live-Vorschau mit Trackbars ---------------------- #
def webcam_segmentierung_mit_trackbars(kamera_index=0):
    cap = cv2.VideoCapture(kamera_index)
    if not cap.isOpened():
        print("❌ Webcam konnte nicht geöffnet werden.")
        return

    cv2.namedWindow("Segmentierung Live")

    # Trackbars erstellen
    cv2.createTrackbar("Cluster", "Segmentierung Live", 6, 12, lambda x: None)
    cv2.createTrackbar("Farbschwelle", "Segmentierung Live", 10, 100, lambda x: None)

    print("🎥 ESC zum Beenden – Regler verändern die Analyse live.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame konnte nicht gelesen werden.")
            break

        frame_resized = cv2.resize(frame, (640, 480))

        clusteranzahl = max(2, cv2.getTrackbarPos("Cluster", "Segmentierung Live"))
        farbschwelle = cv2.getTrackbarPos("Farbschwelle", "Segmentierung Live")

        segmentierungsgrad, clusterbild = berechne_segmentierungsgrad_mit_farbschwelle(
            frame_resized, anzahl_cluster=clusteranzahl, farbschwelle=farbschwelle
        )

        # Original mit Text
        anzeige = frame_resized.copy()
        cv2.putText(
            anzeige, f"Segmentierung: {segmentierungsgrad:.2f}", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4, cv2.LINE_AA
        )

        kombiniert = np.hstack((anzeige, clusterbild))
        cv2.imshow("Segmentierung Live", kombiniert)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- Starten ---------------------- #
if __name__ == "__main__":
    webcam_segmentierung_mit_trackbars()
