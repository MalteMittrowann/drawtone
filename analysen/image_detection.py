import cv2
import pytesseract
import os

# Optional: Tesseract-Pfad für Windows angeben
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\fhs52765\AppData\Local\Programs\Tesseract-OCR'

def erkenne_text(image):
    """Erkennt Text im übergebenen Bild."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='eng+deu')  # Sprachen anpassbar
    return text.strip()

def erkenne_gesichter(image):
    """Erkennt Gesichter im übergebenen Bild und gibt Anzahl zurück."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Absoluter Pfad zur haarcascade-Datei
    haarcascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')

    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    gesichter = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(gesichter), gesichter