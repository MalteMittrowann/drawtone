import cv2

# Kamera starten
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Setze manuelle Belichtungssteuerung, falls unterstützt
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manuell, 0.75 = automatisch

# Anfangswerte festlegen
exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
contrast = cap.get(cv2.CAP_PROP_CONTRAST)

print("Initialwerte:")
print(f"  ➤ Exposure: {exposure}")
print(f"  ➤ Brightness: {brightness}")
print(f"  ➤ Contrast: {contrast}")

# Schrittweite
step = 1

print("\nTasten zur Steuerung:")
print("  W/S: Exposure +/-")
print("  A/D: Brightness +/-")
print("  Q/E: Contrast +/-")
print("  ESC: Beenden\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Zugriff auf die Kamera.")
        break

    # Livebild anzeigen
    cv2.imshow("Live-Kalibrierung (ESC zum Beenden)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('w'):
        exposure += step
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    elif key == ord('s'):
        exposure -= step
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    elif key == ord('a'):
        brightness -= step
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    elif key == ord('d'):
        brightness += step
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    elif key == ord('q'):
        contrast -= step
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    elif key == ord('e'):
        contrast += step
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    elif key == ord('x'):
        # Anzeigen der aktuellen Werte
        print(f"\rExposure: {cap.get(cv2.CAP_PROP_EXPOSURE):>6.2f} | "
            f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS):>6.2f} | "
            f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST):>6.2f}", end='')



cap.release()
cv2.destroyAllWindows()