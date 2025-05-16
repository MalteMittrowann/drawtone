import cv2
import numpy as np

def berechne_abweichung_von_weiss(bgr):
    ziel = np.array([255, 255, 255])
    return np.linalg.norm(ziel - np.array(bgr))

def mittlerer_bgr_wert(frame):
    h, w, _ = frame.shape
    roi = frame[h//2-50:h//2+50, w//2-50:w//2+50]
    return cv2.mean(roi)[:3]

def apply_tint_shift(image, shift):
    # Tint: negative shift = mehr grün, positive shift = mehr magenta
    b, g, r = cv2.split(image)
    r = cv2.add(r, shift)
    g = cv2.add(g, -shift)
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    return cv2.merge((b, g, r))

# Kamera öffnen
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

beste_abweichung = float('inf')
beste_temp = None
bester_tint = None
bestes_frame = None

# Temperatur- und Tint-Werte testen
for temp in range(2800, 7000, 200):
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temp)
    cv2.waitKey(150)
    
    ret, frame = cap.read()
    if not ret:
        continue

    for tint in range(-30, 31, 10):  # Tint-Shift: grün < 0 < magenta
        getönt = apply_tint_shift(frame, tint)
        bgr = mittlerer_bgr_wert(getönt)
        abw = berechne_abweichung_von_weiss(bgr)

        print(f"Temp {temp}, Tint {tint}: Abweichung = {abw:.2f}, BGR = {tuple(int(c) for c in bgr)}")

        if abw < beste_abweichung:
            beste_abweichung = abw
            beste_temp = temp
            bester_tint = tint
            bestes_frame = getönt.copy()

cap.release()

print("\n=====================")
print(f"Beste Temperatur: {beste_temp}")
print(f"Bester Tint-Shift: {bester_tint}")
print(f"Minimale Abweichung: {beste_abweichung:.2f}")
print("=====================\n")

if bestes_frame is not None:
    cv2.imshow("Bestes Ergebnis (kalibriert)", bestes_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
