import cv2
import numpy as np

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

# ---------------------- Live Webcam ---------------------- #

def webcam_farbharmonie_vorschau(kamera_index=0):
    cap = cv2.VideoCapture(kamera_index)
    if not cap.isOpened():
        print("❌ Webcam konnte nicht geöffnet werden.")
        return

    print("🎥 Drücke ESC zum Beenden.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame konnte nicht gelesen werden.")
            break

        frame_resized = cv2.resize(frame, (640, 480))

        # Harmonie und Farbvisualisierung
        harmonie, farbbalken = berechne_farbharmonie(frame_resized)

        # Linke Ansicht: Livebild mit Text
        anzeige = frame_resized.copy()
        cv2.putText(
            anzeige, f"Harmonie: {harmonie:.2f}", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4, cv2.LINE_AA
        )

        # Rechte Ansicht: Farbbalken
        farbbalken_resized = cv2.resize(farbbalken, (anzeige.shape[1], anzeige.shape[0]))

        # Kombinieren und anzeigen
        kombiniert = np.hstack((anzeige, farbbalken_resized))
        cv2.imshow("Farbharmonie Live + Analyse", kombiniert)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC zum Beenden
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- Start ---------------------- #

if __name__ == "__main__":
    webcam_farbharmonie_vorschau()
