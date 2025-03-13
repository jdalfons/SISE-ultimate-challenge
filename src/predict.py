import torch
import torchaudio
import soundfile as sf
import numpy as np
from model.emotion_classifier import EmotionClassifier
from model.feature_extrator import feature_extractor, processor
from utils.preprocessing import resampler
from config import DEVICE, LABELS
import os


# Charger le modèle sauvegardé
classifier = EmotionClassifier(feature_extractor.config.hidden_size, len(LABELS)).to(DEVICE)
classifier.load_state_dict(torch.load(os.path.join("best_emotion_model.pth"), map_location=torch.device(DEVICE)))
classifier.eval()


# Fonction de prédiction
def predict_emotion(audio_path):
    # Charger l'audio
    speech, sample_rate = sf.read(audio_path, dtype="float32")

    # Rééchantillonnage si nécessaire
    if sample_rate != 16000:
        speech = torch.tensor(speech).unsqueeze(0)
        speech = resampler(speech).squeeze(0).numpy()

    # Extraire les features
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        features = feature_extractor(input_values).last_hidden_state.mean(dim=1)
        logits = classifier(features)

    # Obtenir la prédiction
    predicted_label = torch.argmax(logits, dim=-1).item()
    emotion = list(LABELS.keys())[predicted_label]

    return emotion

# Exemple d'utilisation
if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    audio_file = os.path.join(base_path, "colere", "c1ac.wav")
    emotion = predict_emotion(audio_file)
    print(f"🎤 L'émotion prédite est : {emotion}")
