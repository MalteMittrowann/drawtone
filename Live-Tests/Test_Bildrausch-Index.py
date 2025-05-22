import cv2
import numpy as np

# ------------------------- Bildrausch-Index -------------------------- #
def berechne_bildrausch_index(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    varianz = np.var(laplacian)
    index = np.tanh(varianz / 500)  # Normierung auf ca. 0–1
    return index, laplacian

# ------------------------- Live-Vorschau -------------------------- #
def live_bildrausch_debug(kamera_index=0):
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

        # Analyse
        index, laplacian = berechne_bildrausch_index(frame_resized)
        index_text = f"Bildrausch-Index: {index:.3f}"

        # Laplacian visualisieren (Normierung)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        laplacian_color = cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR)

        # Text ins Bild einfügen
        cv2.putText(frame_resized, index_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3, cv2.LINE_AA)

        # Beide Ansichten nebeneinander anzeigen
        kombinierte_ansicht = np.hstack((frame_resized, laplacian_color))
        cv2.imshow("Bildrausch-Analyse (Live)", kombinierte_ansicht)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------- Start -------------------------- #
if __name__ == "__main__":
    live_bildrausch_debug()
