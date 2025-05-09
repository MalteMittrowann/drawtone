import cv2

# Liste gängiger OpenCV-Backends
backends = [
    (cv2.CAP_ANY, "CAP_ANY (Default)"),
    (cv2.CAP_DSHOW, "CAP_DSHOW (DirectShow - Windows empfohlen)"),
    (cv2.CAP_MSMF, "CAP_MSMF (Media Foundation)"),
    (cv2.CAP_V4L2, "CAP_V4L2 (Linux Video4Linux2)"),
    (cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION (macOS)"),
]

def test_backends():
    for api, name in backends:
        print(f"\n🔍 Teste Backend: {name}")
        cap = cv2.VideoCapture(0, api)
        if not cap.isOpened():
            print("❌ Kamera konnte nicht geöffnet werden.")
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Kamera geöffnet, Bildgröße: {frame.shape}")
        else:
            print("⚠️  Kein Bild empfangen (Frame leer oder Fehler).")
        cap.release()

if __name__ == "__main__":
    test_backends()