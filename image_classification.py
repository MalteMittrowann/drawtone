import torch
from PIL import Image
import torchvision.transforms as T
import clip
import cv2

# Lade das CLIP-Modell
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Liste von möglichen Bildbeschreibungen
beschreibungen = [
    # 1 – Sehr harmonisch, ruhig, ästhetisch
    "blank",
    "empty",
    "white canvas",
    "minimal",

    # 2 – Harmonisch mit Energie
    "peaceful",
    "harmonious",
    "calm",
    "soothing",
    "balanced",
    "relaxing",
    "meditative",
    "elegant",
    "vivid",
    "expressive",
    "dynamic",
    "energetic",
    "structured",
    "colorful composition",

    # 3 – Neutral, technisch, durchschnittlich
    "average",
    "neutral",
    "schematic",
    "technical",
    "basic",
    "unremarkable",

    # 4 – Leicht unruhig, erste Dissonanz
    "clashing",
    "uneven",
    "tense",
    "chaotic sketch",
    "disharmony",

    # 5 – Deutlich unruhig, stressig


    # 6 – Unästhetisch, visuell störend
    # 7 – Extrem negativ, verstörend
    "stressful",
    "chaotic",
    "overwhelming",
    "disorganized",
    "visual noise",
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