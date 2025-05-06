import cv2
import numpy as np

def berechne_durchschnittshelligkeit(image):
    # In Graustufen umwandeln
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Durchschnitt berechnen
    return np.mean(gray)