import cv2

def find_working_cameras(max_id=5):
    print("🔍 Suche nach funktionierenden Kamera-IDs...")
    for cam_id in range(max_id):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)  # Direkt DSHOW verwenden
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Kamera {cam_id} funktioniert. Bildgröße: {frame.shape}")
            else:
                print(f"⚠️ Kamera {cam_id} geöffnet, aber kein Bild erhalten.")
            cap.release()
        else:
            print(f"❌ Kamera {cam_id} konnte nicht geöffnet werden.")

find_working_cameras()