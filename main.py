from pythonosc.udp_client import SimpleUDPClient
import cv2
from datetime import datetime
import os
import time
from image_analysis import berechne_durchschnittshelligkeit, berechne_farbanteile, berechne_segmentierungsgrad, berechne_frequenz_index

# OSC-Ziel konfigurieren
osc_ip = "192.168.1.100"  # <-- hier die IP-Adresse des Empfänger-Computers eintragen
osc_port = 8000
client = SimpleUDPClient(osc_ip, osc_port)

def sende_osc_daten(helligkeit, farbanteile):
    client.send_message("/helligkeit", float(helligkeit))
    for farbe, anteil in farbanteile.items():
        client.send_message(f"/farbe/{farbe}", float(anteil))

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #----------------------- Kamera-Kalibierung -----------------------------------#
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.0)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    print("Drücke [Leertaste], um ein Bild aufzunehmen. Drücke [q], um zu beenden.")

    #-------------------------- Aufnahme starten mit Vorschau ----------------------------#
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Bild erhalten.")
            break

        cv2.imshow("Live-Vorschau", frame)
        key = cv2.waitKey(1)

    #------------------------------- Bild abspeichern ------------------------------------#
        if key == ord(" "):  # Leertaste gedrückt
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Basisverzeichnis (Verzeichnis, in dem main.py liegt)
            basisverzeichnis = os.path.dirname(os.path.abspath(__file__))

            # Zielordner relativ zu main.py
            ordner = os.path.join(basisverzeichnis, "captured_images", "tests")
            os.makedirs(ordner, exist_ok=True)

            dateiname = os.path.join(ordner, f"aufnahme_{timestamp}.jpg")
            cv2.imwrite(dateiname, frame)
            print(f"📸 Bild gespeichert als: {dateiname}")

    #--------------------------- Bild-Analyse -------------------------------------#
            helligkeit = berechne_durchschnittshelligkeit(frame)
            farbanteile = berechne_farbanteile(frame)

            print(f"Durchschnittliche Helligkeit: {helligkeit:.2f}")
            print("🎨 Farbanteile:")
            for farbe, anteil in farbanteile.items():
                print(f"  {farbe}: {anteil:.3f}")

            sende_osc_daten(helligkeit, farbanteile)

        elif key == ord("q"):
            print("Programm beendet.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
