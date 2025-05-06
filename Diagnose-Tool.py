import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Versuche automatische Belichtung zu deaktivieren
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = "manual mode" für DirectShow

# Manuelle Belichtung und Helligkeit setzen
cap.set(cv2.CAP_PROP_EXPOSURE, -5)         # Werte zwischen -1 (hell) und -13 (dunkel), je nach Kamera
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)      # Optional: Versuch, Helligkeit zu erhöhen

ret, frame = cap.read()
cap.release()

if ret and frame is not None:
    mean_brightness = frame.mean()
    print(f"📷 Bildgröße: {frame.shape}, Ø-Helligkeit: {mean_brightness:.2f}")
    cv2.imshow("Aufgenommenes Bild", frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
else:
    print("❌ Kein Bild empfangen.")
