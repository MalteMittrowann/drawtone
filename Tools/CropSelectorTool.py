import cv2
import json

def crop_selector_from_webcam(output_json="crop_coords.json"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam konnte nicht geöffnet werden.")
        return

    roi = []
    selected_frame = None

    def select_roi(event, x, y, flags, param):
        nonlocal roi, selected_frame

        if event == cv2.EVENT_LBUTTONDOWN:
            roi = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            roi.append((x, y))
            x1, y1 = roi[0]
            x2, y2 = roi[1]
            cv2.rectangle(selected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Webcam - Ziehe ein Rechteck auf", selected_frame)

            # Koordinaten sortieren
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x1 - x2), abs(y1 - y2)

            # Speichern als JSON
            coords = {"x": x, "y": y, "width": w, "height": h}
            with open(output_json, "w") as f:
                json.dump(coords, f, indent=4)

            print("✅ Crop-Koordinaten gespeichert in:", output_json)

    cv2.namedWindow("Webcam - Ziehe ein Rechteck auf")
    cv2.setMouseCallback("Webcam - Ziehe ein Rechteck auf", select_roi)

    print("📌 Warte auf Auswahl – klicke und ziehe ein Rechteck auf dem Livebild der Webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Fehler beim Lesen des Webcam-Bildes.")
            break

        selected_frame = frame.copy()
        cv2.imshow("Webcam - Ziehe ein Rechteck auf", selected_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# Beispielhafte Verwendung:
if __name__ == "__main__":
    crop_selector_from_webcam()
