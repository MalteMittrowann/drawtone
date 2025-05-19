from pythonosc.udp_client import SimpleUDPClient
import cv2
from datetime import datetime
import os
import numpy as np
import time
from image_analysis import berechne_durchschnittshelligkeit, berechne_farbanteile, berechne_segmentierungsgrad, berechne_frequenz_index, berechne_farbharmonie, berechne_bildrausch_index
from image_classification import klassifiziere_bild_clip, bestimme_genre_wert
from image_detection import erkenne_text, erkenne_gesichter
from projection import projection

#----------------------------- OSC-Send-Modul -------------------------------------#
osc_ip = "78.104.153.86"  # <-- hier die IP-Adresse des Empfänger-Computers eintragen
osc_port = 8000
client = SimpleUDPClient(osc_ip, osc_port)

#-------------------------- Kamera-Kalibrierung ----------------------------------#

# Startwerte für Kamera-Parameter
exposure = -7.5
brightness = 0.0
contrast = 50.0
temp = 3000
tint_shift = 0.0
auto_wb = False

# Startwerte für Morph-Time
morphtime = 60  # Startwert in Sekunden
morphtime_min = 20
morphtime_max = 120
morphtime_step = 1  # Schrittgröße pro Tastendruck

def apply_settings(cap):

    #---- Belichtung ----#
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if not auto_wb else 0.75)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)

    #---- Weißabgleich ----#
    cap.set(cv2.CAP_PROP_AUTO_WB, int(auto_wb))
    cap.set(cv2.CAP_PROP_TEMPERATURE, temp)

def apply_tint(image, tint_shift):
    """
    tint_shift < 0 → grünlicher Tint
    tint_shift > 0 → magentafarbener Tint
    """
    image = image.astype(np.float32)
    image[:, :, 1] *= 1 - abs(tint_shift)  # Grünkanal reduzieren
    if tint_shift > 0:
        image[:, :, 0] *= 1 + tint_shift  # Blau verstärken
        image[:, :, 2] *= 1 + tint_shift  # Rot verstärken
    elif tint_shift < 0:
        image[:, :, 1] *= 1 + abs(tint_shift)  # Grün verstärken
        image[:, :, 0] *= 1 - abs(tint_shift)  # Blau senken
        image[:, :, 2] *= 1 - abs(tint_shift)  # Rot senken
    return np.clip(image, 0, 255).astype(np.uint8)

def finde_optimalen_weissabgleich(cap, thresholdWhite=75, thresholdBlack=25, schritte_temp=100, schritte_tint=0.025): # TODO: Vor dem CR auf schritte_temp=100 & schritte_tint=0.025 resetten
    """
    Findet die Kombination aus Temperatur und Tint, bei der der Weißanteil am höchsten ist.
    """
    best_score = -1
    best_temp = None
    best_tint = None

    global temp, tint_shift  # Nutzt die globale Variable für Temperatur
    original_temp = temp  # Sicherung
    original_tint = tint_shift   # Annahme: Ausgangstint ist 0.0

    # Temperatur-Bereich (typisch: 2800–6500 K)
    for temp_candidate in range(2600, 3501, schritte_temp): # TODO: Vor CR auf 2800 - 6501 resetten
        temp = temp_candidate
        apply_settings(cap)

        # Einige Frames überspringen, damit sich Parameter setzen können
        for _ in range(5):
            ret, _ = cap.read()

        # Tint-Werte im Bereich ±0.5 (entspricht starker Verschiebung)
        for tint_candidate in np.arange(-0.75, 0.251, schritte_tint): # TODO: Vor CR auf - 1.5 bis 1.501 resetten
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Kopie für Analyse in kleiner Auflösung
            analysis_frame = cv2.resize(frame, (320, 240))

            # Tint anwenden
            frame_tinted = apply_tint(analysis_frame, tint_candidate)

            # Farbanteile berechnen
            anteile = berechne_farbanteile(frame_tinted, thresholdWhite, thresholdBlack)
            weiss = anteile.get("weiß", 0)

            print(f"T: {temp_candidate}, Tint: {tint_candidate:.2f} → Weißanteil: {weiss:.3f}")

            if weiss > best_score:
                best_score = weiss
                best_temp = temp_candidate
                best_tint = tint_candidate

    # Ergebnis
    print(f"\n✅ Beste Einstellung: Temp = {best_temp}, Tint = {best_tint:.2f}, Weißanteil = {best_score:.3f}")

    # Ursprungswerte wiederherstellen (optional)
    temp = original_temp
    apply_settings(cap)

    return best_temp, best_tint

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

#-------------------------------- Main-Function -------------------------------------#
def main():
    global exposure, brightness, contrast, temp, auto_wb, tint_shift, morphtime, morphtime_min, morphtime_max, morphtime_step

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    #---- Format ----#
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    # ----- EINMALIGE Kalibrierung zu Beginn -----
    beste_temp, beste_tint = finde_optimalen_weissabgleich(cap)

    # Beste Werte setzen
    temp = beste_temp
    tint_shift = beste_tint

    # Kamera-Kalibrierung starten
    apply_settings(cap)

    cv2.namedWindow("Morph-Time-Vorschau", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Morph-Time-Vorschau", 400, 200)

    print("Druecke LEERTASTE für Bildaufnahme, ESC zum Beenden.")
    print("W/S: Exposure | E/D: Brightness | R/F: Contrast | T/G: Farbtemperatur | Z/H: Tint-Anpassen | A: Auto-WB")

    #-------------------------- Aufnahme starten mit Vorschau ----------------------------#
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Bild erhalten.")
            break
        
        # Kopie für Analyse in kleiner Auflösung
        analysis_frame = cv2.resize(frame, (320, 240))

        #----------------- WB-Tint-Shift ------------------#
        # Tint anwenden
        frame_tinted_analyse = apply_tint(analysis_frame, tint_shift)
        frame_tinted = apply_tint(frame, tint_shift)

        #-------------- Video-Voschau starten --------------#
        cv2.imshow("Live-Vorschau, ESC druecken zum Beenden", frame_tinted)
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
            temp = min(temp + 200, 20000) # type: ignore
            print(f"WB-Temp: {temp:.2f}")
        elif key == ord('g'):
            temp = max(temp - 200, 0) # type: ignore
            print(f"WB-Temp: {temp:.2f}")
        elif key == ord('z'):  # Mehr Magenta
            tint_shift = min(tint_shift + 0.05, 0.5) # type: ignore
            print(f"Tint: {tint_shift}")
        elif key == ord('h'):  # Mehr Grün
            tint_shift = max(tint_shift - 0.05, -0.5) # type: ignore
            print(f"Tint: {tint_shift}")
        elif key == ord('a'):
            auto_wb = not auto_wb
            print(f"Auto Weißabgleich: {'AN' if auto_wb else 'AUS'}")
        elif key == ord('u'):  # Erhöhe Morph-Time
            morphtime = min(morphtime + morphtime_step, morphtime_max)
            print(f"↑ Morph-Time: {morphtime} Sek.")
        elif key == ord('j'):  # Verringere Morph-Time
            morphtime = max(morphtime - morphtime_step, morphtime_min)
            print(f"↓ Morph-Time: {morphtime} Sek.")

    #------------------------------- Bild abspeichern ------------------------------------#
        elif key == 32:  # Leertaste
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Basisverzeichnis (Verzeichnis, in dem main.py liegt)
            basisverzeichnis = os.path.dirname(os.path.abspath(__file__))

            # Zielordner relativ zu main.py
            ordner = os.path.join(basisverzeichnis, "captured_images", "tests")
            os.makedirs(ordner, exist_ok=True)

            dateiname = os.path.join(ordner, f"aufnahme_{timestamp}.jpg")
            cv2.imwrite(dateiname, frame_tinted)
            print(f"📸 Bild gespeichert als: {dateiname}")

    #--------------------------- Bild-Analyse -------------------------------------#
            anzahl_cluster = 6

            client.send_message("/morphtime", morphtime)

            #---------------- Helligkeit ----------------#
            helligkeit = berechne_durchschnittshelligkeit(frame_tinted_analyse)

            helligkeit_gemappt = map_value(helligkeit, 0, 255, 20, 600)
            print(f"Durchschnittliche Helligkeit: {helligkeit:.2f}, Gemappte Hellgkeit: {helligkeit_gemappt:.2f}")

            client.send_message("/grundton", float(helligkeit_gemappt))

            #---------------- Farbanalyse ---------------#
            farbanteile = berechne_farbanteile(frame_tinted_analyse, 75, 25)
            print("🎨 Farbanteile:")
            for farbe, anteil in farbanteile.items():
                print(f"  {farbe}: {anteil:.3f}")

            #--- Farbanteile mappen und senden ---#
            for farbe, anteil in farbanteile.items():
                if anteil < 0.025:
                    farbanteile_mapped = 0.0
                elif anteil >= 1.0:
                    farbanteile_mapped = 1.0
                else:
                    farbanteile_mapped = map_value(anteil, 0.025, 1.0, 0.7, 1.0)

                client.send_message(f"/{farbe}", float(farbanteile_mapped))

            #------------ Segmentierungsgrad ------------#
            segmentierungsgrad = berechne_segmentierungsgrad(frame_tinted_analyse, anzahl_cluster)
            print(f"Segmentierungs-Grad: {segmentierungsgrad:.2f} | Einfarbig/flächig (gering segmentiert) | bunt/kleinteilig (hoch segmentiert)")
            print("🧭 Interpretation der Werte: ~ 0.0 – 0.2	Sehr gleichmäßige Clustergrößen → Bild hat gleichmäßig verteilte Farben | ~ 0.2 – 0.5	Mäßige Unterschiede in der Flächenverteilung | ~ 0.5 – 1.0+	Einige Cluster dominieren → starke farbliche Fragmentierung oder viele kleine Details")
            
            segmentierungsgradClamped = max(0.0, min(1.0, segmentierungsgrad))
            client.send_message("/segmentierungsgrad", segmentierungsgradClamped)

            #------------- Frequenz-Index --------------#
            frequenz_index = berechne_frequenz_index(frame_tinted_analyse)
            print(f"Frequenz-Index: {frequenz_index:.2f} | Niedrige Frequenzen → große, flächige Strukturen (ruhige Bilder, wenig Details) | Hohe Frequenzen → viele Kanten, feine Details, Muster (z. B. Kritzeleien, Texturen, Rauschen)")
            print("🧭 Interpretation der Werte: < 0.1	Sehr flächig, fast keine feinen Details | 0.1 – 0.5	Eher ruhig, moderate Details | 0.5 – 1.0	Ausgewogen zwischen Fläche und Detail | > 1.0	Viele feine Details, starke Kanten, „wilde“ Bildstruktur | > 2.0 – 5.0	Sehr detailreich oder rauschig")
            
            frequenz_index_mapped = map_value(frequenz_index, 0, 10, 0, 127)
            frequenz_index_mapped_clamped = max(0.0, min(127.0, frequenz_index_mapped))
            client.send_message("/drumsample", frequenz_index_mapped_clamped)

            #-------------- Farbharmonie ---------------#
            farbharmonie = berechne_farbharmonie(frame_tinted_analyse, anzahl_cluster)
            print(f"Farbharmonie: {farbharmonie:.2f} | Große Abstände = starke Kontraste → „unharmonisch“ | Kleine Abstände = ähnliche Farben → „harmonisch“")
            print("🧠 Interpretation des Werts: 1.0 → Sehr harmonisch (ähnliche Farben) | 0.0 → Sehr kontrastreich (komplementäre Farben)")

            #-------------- Bildrauschen ---------------#
            bildrauschen_index, bildrauschen_varianz = berechne_bildrausch_index(frame_tinted_analyse)
            print(f"Bildrauschen-Index: {bildrauschen_index:.2f} | Bildrauschen_Varianz: {bildrauschen_varianz: .2f} | Viele Kanten und hohe Bildfrequenzen = „visuelle Unruhe“")
            print("📊 Typische Werte: 0.0 – 0.2: Sehr glatt, kaum Details | 0.3 – 0.6: Mittlere Textur, normale Bilder | 0.7 – 1.0: Sehr detailreich oder visuell überladen")

            #---------- Bild-Kategorisierung -----------#
            top3_Kategorien = klassifiziere_bild_clip(frame_tinted_analyse)
            print("→ KI-Analyse (Top 3 Kategorien):")
            for beschreibung, score in top3_Kategorien:
                print(f"  - {beschreibung}: {score:.2%}")
            
            # Genre bestimmen und per OSC senden
            genre_wert = bestimme_genre_wert(top3_Kategorien)
            client.send_message("/genre", genre_wert)

            print(f"Gesendeter Genrewert: {genre_wert:.2f} --> von 1-7")

            client.send_message("/BPM", 180)

            time.sleep(0.2)

            client.send_message("/morph", 1)
            print("Abfahrt!")

    #------------------------------ Projektion -------------------------------------#
            analysis_text = [
                f"Bildrauschen-Varianz: {bildrauschen_varianz:.2f}",
                f"Bildrauschen-Index: {bildrauschen_index:.2f}",
                f"Farbharmonie: {farbharmonie:.2f}",
                f"Frequenzindex: {frequenz_index:.2f}",
                f"Segmentierungsgrad: {segmentierungsgradClamped:.2f}",
                f"Grundton: {helligkeit_gemappt:.2f}",
                f"Helligkeit: {helligkeit:.2f}"
            ]
            #------- Projektion starten -------#
            # Bild mit Analyse anzeigen (Projektion)
            projection(frame_tinted, analysis_text, 10.0, 10)

    #-------------------------- Bild-Erkennung -------------------------------------#
            #text = erkenne_text(frame_tinted_analyse)
            #print("Erkannter Text:", text)
            #anzahl_gesichter, gesichter = erkenne_gesichter(frame_tinted_analyse)
            #print(f"Anzahl erkannter Gesichter: {anzahl_gesichter}")

        # Nach Tastenänderung Settings erneut anwenden
        apply_settings(cap)

    #----------------------------- Morphtime-Werte anzeigen ------------------------------#

        # Vorschau-Bild erzeugen
        preview_img = np.zeros((200, 400, 3), dtype=np.uint8)
        text = f"Morph-Time: {morphtime}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (preview_img.shape[1] - text_size[0]) // 2
        text_y = (preview_img.shape[0] + text_size[1]) // 2

        cv2.putText(preview_img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.imshow("Morph-Time-Vorschau", preview_img)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
