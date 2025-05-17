import cv2
from screeninfo import get_monitors

def projection(image):
    # Bildschirm-Infos holen
    monitore = get_monitors()
    if len(monitore) < 2:
        print("⚠️ Nur ein Bildschirm erkannt.")
        x, y = 0, 0  # Default
    else:
        # Zweiter Monitor (meistens index 1)
        monitor = monitore[1]
        x, y = monitor.x, monitor.y
        print(f"📺 Zweitmonitor: Position = ({x}, {y}), Größe = {monitor.width}x{monitor.height}")

    bild = image.copy()

    cv2.namedWindow("Projektion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Projektion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Projektion", x, 0)  # → Position für 2. Bildschirm anpassen!
    cv2.imshow("Projektion", bild)
    cv2.waitKey(1)

def projectionMitHelligkeit(image, helligkeit):
    bild = image.copy()

        # Text aufs Bild schreiben
    cv2.putText(
        bild,
        f"Helligkeit: {helligkeit:.2f}",
        org=(50, 100),  # Position (x, y)
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(255, 255, 255),  # Weiß
        thickness=4,
        lineType=cv2.LINE_AA
    )

    cv2.namedWindow("Projektion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Projektion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Projektion", 2880, 0)  # → Position für 2. Bildschirm anpassen!
    cv2.imshow("Projektion", bild)
    cv2.waitKey(1)

def projectionMitHelligkeit_Farben(image, helligkeit, farbanteile):
    bild = image.copy()

    cv2.namedWindow("Projektion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Projektion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Projektion", 2880, 0)  # → Position für 2. Bildschirm anpassen!
    cv2.imshow("Projektion", bild)
    cv2.waitKey(1)