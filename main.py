from datetime import datetime
import cv2
from image_analysis import berechne_durchschnittshelligkeit, berechne_farbanteile
import time
import os

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # (int ID, int apiPreference)

    #----------------------- Kamera-Kalibierung -----------------------------------#
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # 0.25 = manuell (abhängig vom Backend), 0.75 = automatisch
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0) # Kalibriert auf -6.0
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.0) # Kalibiert auf 0.0
    cap.set(cv2.CAP_PROP_CONTRAST, 32) # Kalibriert auf 32

    # Dimensionen des Fotos einrichten:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    #------------------------ Aufnahme vorbereiten --------------------------------#
    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    # 1–2 Sekunden warten, damit die Kamera sich anpassen kann
    time.sleep(2)

    # Einige Frames "puffern" (Kamera justieren lassen)
    for _ in range(10):
        cap.read()

    #------------------------- Live-Vorschau --------------------------------------#
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

    #-------------------------- Bild abspeichern ----------------------------------#
    # Ordnerpfad definieren
    ordner = "captured_images/tests"
    os.makedirs(ordner, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden

    # Zeitstempel erzeugen
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dateiname = os.path.join(ordner, f"aufnahme_{timestamp}.jpg")

    # Bild abspeichern mit Zeitstempel
    cv2.imwrite(dateiname, frame)
    print(f"Bild gespeichert als: {dateiname}")

    #--------------------------- Bild-Analyse -------------------------------------#
    # Bild analysieren
    helligkeit = berechne_durchschnittshelligkeit(frame)
    print(f"Durchschnittliche Helligkeit: {helligkeit}")

    # Farbanteile berechnen
    farbanteile = berechne_farbanteile(frame)
    print("🎨 Farbanteile:")
    for farbe, anteil in farbanteile.items():
        print(f"  {farbe}: {anteil:.3f}")

    #--------------------------- Main-Program -------------------------------------#
if __name__ == "__main__":
    main()
