#-----------------------------------------------------------
 # DRAWTONE
 # Copyright (c) 2026 Dave Kronawitter & Malte Mittrowann.
 # All rights reserved.
 #
 # This code is proprietary and not open source.
 # Unauthorized copying of this file is strictly prohibited.
#------------------------------------------------------------


import cv2
import numpy as np

# ------------------------- Bildrausch-Index -------------------------- #
def visualisiere_bildrausch(image):
    vis = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_8U)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    #cv2.putText(vis, f"Rausch: {index:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    return vis

# ------------------------- Farbschwerpunkt -------------------------- #
def visualisiere_farbschwerpunkt(image, farbschwerpunkt):
    # Form umwandeln von (3,) → (1, 1, 3) für cv2.cvtColor
    farbe = np.uint8([[farbschwerpunkt]])  # type: ignore # 1x1 HSV-Feld
    farbe_rgb = cv2.cvtColor(farbe, cv2.COLOR_HSV2BGR)[0, 0] # type: ignore
    
    # Einfarbiges Bild erzeugen in der errechneten Farbe
    vis = np.full_like(image, farbe_rgb)
    #cv2.putText(vis, f"Farbzentrum: H={int(h)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return vis

# ------------------------- Frequenzanalyse -------------------------- #
def visualisiere_frequenzanalyse(spectrum):
    norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
    vis = cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #cv2.putText(vis, "Frequenzanalyse", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 100), 3)
    return vis

# ------------------------- Farbanteile -------------------------- #
def berechne_farbanteile(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    farben = hsv[:, :, 0].flatten()
    hist = cv2.calcHist([farben], [0], None, [12], [0, 180])
    hist_norm = hist / hist.sum()
    return hist_norm

def visualisiere_farbanteile(image):
    hist = berechne_farbanteile(image)
    breite = 50
    hoehe = 400 # Hoehe einstellbar
    bild = np.zeros((hoehe, breite * len(hist), 3), dtype=np.uint8)
    for i, h in enumerate(hist):
        farbe = np.uint8([[[i * 15, 255, 255]]]) # type: ignore
        bgr = cv2.cvtColor(farbe, cv2.COLOR_HSV2BGR)[0, 0].tolist() # type: ignore
        cv2.rectangle(bild, (i * breite, hoehe), ((i + 1) * breite, hoehe - int(h * hoehe)), bgr, -1)
    vis = cv2.resize(bild, (image.shape[1], image.shape[0]))
    #cv2.putText(vis, "Farbanteile", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return vis
