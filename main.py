from datetime import datetime
import cv2
from image_analysis import berechne_durchschnittshelligkeit
import time
import os

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #cap.set(cv2.CAP_PROP_EXPOSURE, 0)
    #cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    #cap.set(cv2.CAP_PROP_CONTRAST, 0)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    ret, frame = cap.read()
    if ret:
        cv2.imshow("Vorschau", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Kein Bild erhalten.")

    cap.release()

    if not ret:
        print("Fehler: Bild konnte nicht aufgenommen werden.")
        return

    # Ordnerpfad definieren
    ordner = "captured_images/tests"
    os.makedirs(ordner, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden

    # Zeitstempel erzeugen
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dateiname = os.path.join(ordner, f"aufnahme_{timestamp}.jpg")

    # Bild abspeichern mit Zeitstempel
    cv2.imwrite(dateiname, frame)
    print(f"Bild gespeichert als: {dateiname}")

    # Bild analysieren
    helligkeit = berechne_durchschnittshelligkeit(frame)
    print(f"Durchschnittliche Helligkeit: {helligkeit}")

if __name__ == "__main__":
    main()
