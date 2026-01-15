#-----------------------------------------------------------
 # DRAWTONE
 # Copyright (c) 2026 Dave Kronawitter & Malte Mittrowann.
 # All rights reserved.
 #
 # This code is proprietary and not open source.
 # Unauthorized copying of this file is strictly prohibited.
#------------------------------------------------------------


import torch
from PIL import Image
import torchvision.transforms as T
import clip
import cv2

# Lade das CLIP-Modell
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#--------------------------- Bild-KI-Analyse --------------------------------#
# Liste von möglichen Bildbeschreibungen
beschreibungen = [
    # 1 – Sehr harmonisch, ruhig, ästhetisch --> Keine Drums!
    "blank",
    "empty",
    "white canvas",

    # 2 – Harmonisch mit Energie --> Kick-Only
    "minimal",
    "basic",
    "peaceful",
    "harmonious",
    "calm",
    "soothing",
    "balanced",
    "relaxing",
    "meditative",
    "elegant",

    # 3 – Neutral, technisch, durchschnittlich --> Trap (Half-Time-Groove)
    "average",
    "neutral",
    "schematic",
    "technical",
    "basic",
    "unremarkable",
    #"colorful composition",

    # 4 – Leicht unruhig, erste Dissonanz --> House
    "vivid",
    "expressive",
    "dynamic",
    "energetic",
    "structured",

    # 5 – Deutlich unruhig, stressig --> Techno
    "geometric",
    "pattern",
    "repetitive",
    "dark",

    # 6 – Unästhetisch, visuell störend --> D & B
    "clashing",
    "uneven",
    "tense",
    "chaotic sketch",
    "disharmony",

    # 7 – Extrem negativ, verstörend --> Random
    "stressful",
    "chaotic",
    "overwhelming",
    "disorganized",

    #"visual noise",
    "ugly",
    "harsh",
    "distorted",
    "unpleasant",
    "painful",
    "violent",
    "aggressive",
    "terrifying",
    "destructive",
    "angry",
    "disturbing"
]

text_tokens = clip.tokenize(beschreibungen).to(device)

#----- Funktion zur KI-Klassifizierung -----#
def klassifiziere_bild_clip(cv2_image):
    # BGR → RGB
    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # In PIL-Bild umwandeln
    pil_img = Image.fromarray(img_rgb)

    # Preprocessen und auf Modell schicken
    image_input = preprocess(pil_img).unsqueeze(0).to(device) # type: ignore

    # Ähnlichkeiten berechnen
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        ähnlichkeiten = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Top 3 Ergebnisse sortieren und zurückgeben
    top3_indices = torch.topk(ähnlichkeiten[0], 3).indices
    top3 = [(beschreibungen[i], ähnlichkeiten[0][i].item()) for i in top3_indices]

    return top3

#--------------------------- Kategorisierung --------------------------------#
# Zuweisung der Analyse Prompts zu Genres ---> Werte von 1 - 7
genre_mapping = {
    # 1 – Sehr harmonisch, ruhig, ästhetisch
    "blank": 1,
    "empty": 1,
    "white canvas": 1,

    # 2 – Harmonisch mit Energie
    "minimal": 2,
    "basic": 2,
    "peaceful": 2,
    "harmonious": 2,
    "calm": 2,
    "soothing": 2,
    "balanced": 2,
    "relaxing": 2,
    "meditative": 2,
    "elegant": 2,

    # 3 – Neutral, technisch, durchschnittlich
    "average": 3,
    "neutral": 3,
    "schematic": 3,
    "technical": 3,
    "basic": 3,
    "unremarkable": 3,
    #"colorful composition": 3,

    # 4 – Leicht unruhig, erste Dissonanz
    "vivid": 4,
    "expressive": 4,
    "dynamic": 4,
    "energetic": 4,
    #"structured": 4,

    # 5 – Deutlich unruhig, stressig
    "geometric": 5,
    "pattern": 5,
    "repetitive": 5,
    "dark": 5,

    # 6 – Unästhetisch, visuell störend
    "clashing": 6,
    "uneven": 6,
    "tense": 6,
    "chaotic sketch": 6,
    "disharmony": 6,

    # 7 – Extrem negativ, verstörend
    #"visual noise": 7,
    "stressful": 7,
    "chaotic": 7,
    "overwhelming": 7,
    "disorganized": 7,

    "ugly": 7,
    "harsh": 7,
    "distorted": 7,
    "unpleasant": 7,
    "painful": 7,

    "violent": 7,
    "aggressive": 7,
    "terrifying": 7,
    "destructive": 7,
    "angry": 7,
    "disturbing": 7
}

#----- Funktion zur Zuwesiung der Prompts zu den Genre-Werten von 1-7 -----#
def bestimme_genre_wert(top3_kategorien):
    """Bestimme Genre-Wert basierend auf dem Beschreibungstext mit höchstem Score."""
    if not top3_kategorien:
        return 3  # Neutraler Fallback

    beste_beschreibung = top3_kategorien[0][0].lower()

    for key, wert in genre_mapping.items():
        if key in beste_beschreibung:
            return wert

    return 3  # Fallback auf neutral