import os
import cv2
import time
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from image_analysis import (
    berechne_durchschnittshelligkeit,
    berechne_farbanteile,
    berechne_segmentierungsgrad,
    berechne_frequenz_index,
    berechne_farbharmonie,
    berechne_bildrausch_index
)
#from image_classifier import klassifiziere_bild_clip

# OSC-Konfiguration
OSC_IP = "172.20.10.14"  # Ziel-IP (localhost)
OSC_PORT = 8000       # Ziel-Port (in PureData oder anderer Software)
osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)

def sende_osc_wert(adresse, wert):
    osc_client.send_message(adresse, wert)

def main():
    # Eingabe des Bildnamens
    dateiname = input("Bitte gib den Dateinamen des zu analysierenden Bildes ein (z. B. 'testbild.jpg'): ").strip()
    pfad = os.path.join("captured_images", "tests", dateiname)

    if not os.path.exists(pfad):
        print("❌ Bildpfad existiert nicht:", pfad)
        return

    # Bild einlesen
    frame = cv2.imread(pfad)
    if frame is None:
        print("❌ Bild konnte nicht geladen werden.")
        return

    print(f"✅ Bild erfolgreich geladen: {pfad}")

    # --------------------------- Bild-Analyse ------------------------------------- #
    anzahl_cluster = 6
    helligkeit = berechne_durchschnittshelligkeit(frame)
    farbanteileRecieve = berechne_farbanteile(frame, 50)
    segmentierungsgrad = berechne_segmentierungsgrad(frame, anzahl_cluster)
    frequenz_index = berechne_frequenz_index(frame)
    farbharmonie = berechne_farbharmonie(frame, anzahl_cluster)
    bildrauschen = berechne_bildrausch_index(frame)

    farbanteileSend = {
        "rot": farbanteileRecieve.get("rot", 0) + farbanteileRecieve.get("magenta", 0),     # Chords
        "grün": farbanteileRecieve.get("grün", 0),                                          # Drums
        "blau": farbanteileRecieve.get("blau", 0) + farbanteileRecieve.get("cyan", 0),      # Bass
        "gelb": farbanteileRecieve.get("gelb", 0),                                          # Melodie
        "weiß": farbanteileRecieve.get("weiß", 0),                                          # Sonstiges
        "schwarz": farbanteileRecieve.get("schwarz", 0)                                     # Sonstiges
    }

    # -------------------------- Bild-Kategorisierung ----------------------------- #
    #top3_Kategorien = klassifiziere_bild_clip(frame)

    # ----------------------------- OSC-Senden ------------------------------------- #
    sende_osc_wert("/grundton", helligkeit) # Helligkeit --> Grundton
    # Segmentierungsgrad clampen und senden
    segmentierungsgrad = max(0.0, min(1.0, segmentierungsgrad))
    sende_osc_wert("/segmentierungsgrad", segmentierungsgrad)
    sende_osc_wert("/frequenz", frequenz_index)
    #sende_osc_wert("/farbharmonie", farbharmonie)
    sende_osc_wert("/bildrauschen", bildrauschen)
    sende_osc_wert("/BPM", 120)
    sende_osc_wert("/morphtime", 5.0)

    time.sleep(1.0)

    sende_osc_wert("/morph", 1)

    # Farbanteile senden (geclamped)
    for farbe, anteil in farbanteileSend.items():
        if anteil < 0.05:
            anteil = 0.0
        elif anteil > 0.3:
            anteil = 1.0
        sende_osc_wert(f"/{farbe}", anteil)

    # Kategorien senden
    #for i, (label, score) in enumerate(top3_Kategorien):
    #    sende_osc_wert(f"/clip/top{i+1}/label", label)
    #    sende_osc_wert(f"/clip/top{i+1}/score", score)

    # ----------------------------- Ausgabe ---------------------------------------- #
    print("\n📊 Analyseergebnisse:")
    print(f"• Durchschnittshelligkeit: {helligkeit:.3f}")
    print(f"• Farbanteile (geclamped):")
    for farbe, anteil in farbanteileRecieve.items():
        clamp = 0.0 if anteil < 0.05 else (1.0 if anteil > 0.3 else anteil)
        print(f"  - {farbe}: {clamp:.2f}")
    print(f"• Segmentierungsgrad: {segmentierungsgrad:.3f}")
    print(f"• Frequenzindex: {frequenz_index:.3f}")
    print(f"• Farbharmonie: {farbharmonie:.3f}")
    print(f"• Bildrausch-Index: {bildrauschen:.3f}")

    #print("\n🧠 Top 3 Kategorien (CLIP):")
    #for label, score in top3_Kategorien:
    #    print(f"  - {label}: {score:.4f}")

if __name__ == "__main__":
    main()
