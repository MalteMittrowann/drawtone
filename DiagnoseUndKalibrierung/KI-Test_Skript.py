import os
from PIL import Image
import torch
import open_clip

# Modell und Preprocessing laden
model, _, preprocesses = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
preprocess = preprocesses  # Eval-Preprocessor
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# GPU nutzen, falls vorhanden
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Liste mit Prompts (Kategorien)
labels = [
    "a child drawing", "a landscape", "a face", "abstract art", "scribbles",
    "shapes", "text", "a tree", "a house", "a person", "a sun", "a cat",
    "a fish", "a flower", "a car", "a mountain", "a rainbow", "a star",
    "nothing", "chaos", "organized pattern"
]
text_tokens = tokenizer(labels).to(device)

# Verzeichnis mit Bildern
ordner_pfad = "./captured_images/tests"

# Jedes Bild analysieren
for dateiname in os.listdir(ordner_pfad):
    if dateiname.lower().endswith(('.jpg', '.jpeg', '.png')):
        pfad = os.path.join(ordner_pfad, dateiname)
        print(f"\n🔍 Bild: {dateiname}")

        # Bild laden und vorbereiten
        image = preprocess(Image.open(pfad).convert("RGB")).unsqueeze(0).to(device) # type: ignore

        # Ähnlichkeiten berechnen
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)

            # Kosinus-Ähnlichkeiten berechnen
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            ähnlichkeiten = (100.0 * image_features @ text_features.T).squeeze(0)

            # Top-3 Kategorien anzeigen
            topk = torch.topk(ähnlichkeiten, k=3)
            for score, index in zip(topk.values, topk.indices):
                print(f"  → {labels[index]:<25} ({score.item():.2f})")