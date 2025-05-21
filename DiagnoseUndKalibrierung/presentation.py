import cv2
import os
from datetime import datetime

# === Konfiguration ===
KAMERA_INDEX = 0  # meistens 0 für die Standardkamera
SPEICHERORDNER = "aufnahmen"
VERWENDE_MSMF = cv2.CAP_MSMF  # Media Foundation (Windows)

# Kamera starten
cap = cv2.VideoCapture(KAMERA_INDEX, VERWENDE_MSMF)

if not cap.isOpened():
    print("❌ Kamera konnte nicht geöffnet werden.")
    exit()

# Kameraeinstellungen setzen (nur wenn von der Kamera unterstützt)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)      # Wertebereich: 0.0–1.0 (je nach Kamera)
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)        # Wertebereich: 0.0–1.0
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4500)  # Farbtemperatur (K), falls unterstützt
cap.set(cv2.CAP_PROP_AUTO_WB, 1)           # Automatischer Weißabgleich aktivieren (1 = an)

print("📷 Kamera gestartet. Drücke 's' zum Speichern eines Bildes oder 'q' zum Beenden.")

# Ordner anlegen, falls nicht vorhanden
os.makedirs(SPEICHERORDNER, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Kein Kamerabild empfangen.")
        break

    cv2.imshow("Live-Vorschau", frame)
    taste = cv2.waitKey(1) & 0xFF

    if taste == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dateiname = f"{SPEICHERORDNER}/bild_{timestamp}.jpg"
        cv2.imwrite(dateiname, frame)
        print(f"✅ Bild gespeichert: {dateiname}")
    elif taste == ord('q'):
        print("👋 Beende Live-Vorschau.")
        break

cap.release()
cv2.destroyAllWindows()
