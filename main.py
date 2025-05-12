from pythonosc.udp_client import SimpleUDPClient
import cv2
from datetime import datetime
import os
import time
from image_analysis import berechne_durchschnittshelligkeit, berechne_farbanteile, berechne_segmentierungsgrad, berechne_frequenz_index, berechne_farbharmonie, berechne_bildrausch_index
from image_classification import klassifiziere_bild_clip
#----------------------------- OSC-Send-Modul -------------------------------------#
osc_ip = "10.40.35.127"  # <-- hier die IP-Adresse des Empfänger-Computers eintragen
osc_port = 8000
client = SimpleUDPClient(osc_ip, osc_port)

def sende_osc_daten(helligkeit, farbanteile):
    client.send_message("/helligkeit", float(helligkeit))
    for farbe, anteil in farbanteile.items():
        client.send_message(f"/farbe/{farbe}", float(anteil))
    client.send_message("/morphtime", 20.0)
    client.send_message("/BPM", 120)
    client.send_message("/genre", 3)
    client.send_message("/morph", 1)

#-------------------------- Kamera-Kalibrierung ----------------------------------#

# Startwerte für Kamera-Parameter
exposure = -6.0
brightness = 0.0
contrast = 32.0
temp = 3000
auto_wb = False

def apply_settings(cap):

    #---- Belichtung ----#
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if not auto_wb else 0.75)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)

    #---- Weißabgleich ----#
    cap.set(cv2.CAP_PROP_AUTO_WB, int(auto_wb))
    cap.set(cv2.CAP_PROP_TEMPERATURE, temp)

#-------------------------------- Main-Function -------------------------------------#
def main():
    global exposure, brightness, contrast, temp, auto_wb

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    #---- Format ----#
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    # Kamera-Kalibrierung starten
    apply_settings(cap)

    print("Druecke LEERTASTE für Bildaufnahme, ESC zum Beenden.")
    print("W/S: Exposure | E/D: Brightness | R/F: Contrast | T/G: Farbtemperatur | A: Auto-WB")

    #-------------------------- Aufnahme starten mit Vorschau ----------------------------#
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Bild erhalten.")
            break

        cv2.imshow("Live-Vorschau, ESC druecken zum Beenden", frame)
        key = cv2.waitKey(1) & 0xFF

    #------------------------- Kamera nachträglich kalibieren ----------------------------#
        # Tastensteuerung
        if key == 27:  # ESC
            print("Beenden!")
            break
        elif key == ord('w'):
            exposure += 0.5
            print(f"Exposure: {exposure:.2f}")
        elif key == ord('s'):
            exposure -= 0.5
            print(f"Exposure: {exposure:.2f}")
        elif key == ord('e'):
            brightness += 1.0
            print(f"Brightness: {brightness:.2f}")
        elif key == ord('d'):
            brightness -= 1.0
            print(f"Brightness: {brightness:.2f}")
        elif key == ord('r'):
            contrast += 1.0
            print(f"Contrast: {contrast:.2f}")
        elif key == ord('f'):
            contrast -= 1.0
            print(f"Contrast: {contrast:.2f}")
        elif key == ord('t'):
            temp = min(temp + 200, 20000)
            print(f"WB-Temp: {temp:.2f}")
        elif key == ord('g'):
            temp = max(temp - 200, 0)
            print(f"WB-Temp: {temp:.2f}")
        elif key == ord('a'):
            auto_wb = not auto_wb
            print(f"Auto Weißabgleich: {'AN' if auto_wb else 'AUS'}")

    #------------------------------- Bild abspeichern ------------------------------------#
        elif key == 32:  # Leertaste
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
            anzahl_cluster = 6
            helligkeit = berechne_durchschnittshelligkeit(frame)
            farbanteile = berechne_farbanteile(frame, 50)
            segmentierungsgrad = berechne_segmentierungsgrad(frame, anzahl_cluster)
            frequenz_index = berechne_frequenz_index(frame)
            farbharmonie = berechne_farbharmonie(frame, anzahl_cluster)
            bildrauschen = berechne_bildrausch_index(frame)
            top3_Kategorien = klassifiziere_bild_clip(frame)

            print(f"Durchschnittliche Helligkeit: {helligkeit:.2f}")
            print("🎨 Farbanteile:")
            for farbe, anteil in farbanteile.items():
                print(f"  {farbe}: {anteil:.3f}")
            print(f"Frequenz-Index: {frequenz_index:.2f} | Niedrige Frequenzen → große, flächige Strukturen (ruhige Bilder, wenig Details) | Hohe Frequenzen → viele Kanten, feine Details, Muster (z. B. Kritzeleien, Texturen, Rauschen)")
            print("🧭 Interpretation der Werte: < 0.1	Sehr flächig, fast keine feinen Details | 0.1 – 0.5	Eher ruhig, moderate Details | 0.5 – 1.0	Ausgewogen zwischen Fläche und Detail | > 1.0	Viele feine Details, starke Kanten, „wilde“ Bildstruktur | > 2.0 – 5.0	Sehr detailreich oder rauschig")
            print(f"Segmentierungs-Grad: {segmentierungsgrad:.2f} | Einfarbig/flächig (gering segmentiert) | bunt/kleinteilig (hoch segmentiert)")
            print("🧭 Interpretation der Werte: ~ 0.0 – 0.2	Sehr gleichmäßige Clustergrößen → Bild hat gleichmäßig verteilte Farben | ~ 0.2 – 0.5	Mäßige Unterschiede in der Flächenverteilung | ~ 0.5 – 1.0+	Einige Cluster dominieren → starke farbliche Fragmentierung oder viele kleine Details")
            print(f"Farbharmonie: {farbharmonie:.2f} | Große Abstände = starke Kontraste → „unharmonisch“ | Kleine Abstände = ähnliche Farben → „harmonisch“")
            print("🧠 Interpretation des Werts: 1.0 → Sehr harmonisch (ähnliche Farben) | 0.0 → Sehr kontrastreich (komplementäre Farben)")
            print(f"Bildrauschen: {bildrauschen:.2f} | Viele Kanten und hohe Bildfrequenzen = „visuelle Unruhe“")
            print("📊 Typische Werte: 0.0 – 0.2: Sehr glatt, kaum Details | 0.3 – 0.6: Mittlere Textur, normale Bilder | 0.7 – 1.0: Sehr detailreich oder visuell überladen")
            print("→ KI-Analyse (Top 3 Kategorien):")
            for beschreibung, score in top3_Kategorien:
                print(f"  - {beschreibung}: {score:.2%}")
            # sende_osc_daten(helligkeit, farbanteile)

        # Nach Tastenänderung Settings erneut anwenden
        apply_settings(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
