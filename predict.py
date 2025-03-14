import sys
import os
import torch
import librosa
import numpy as np
from model.emotion_classifier import EmotionClassifier
from utils.preprocessing import collate_fn
from config import DEVICE, NUM_LABELS, BEST_MODEL_NAME

# Charger le modèle entraîné
feature_dim = 40  # Nombre de MFCCs utilisés
model = EmotionClassifier(feature_dim, NUM_LABELS).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_NAME, map_location=DEVICE))
model.eval()  # Mode évaluation

# Labels des émotions
LABELS = {0: "colère", 1: "neutre", 2: "joie"}

# Fonction pour prédire l’émotion d’un fichier audio avec probabilités
def predict_emotion(audio_path, max_length=128):
    # Charger l’audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Extraire les MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Ajuster la taille des MFCCs avec padding/troncature
    if mfcc.shape[1] > max_length:  
        mfcc = mfcc[:, :max_length]  # Tronquer si trop long
    else:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Convertir en tenseur PyTorch
    input_tensor = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, max_length, 40)

    # Prédiction avec le modèle
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().flatten()  # Convertir en probabilités
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Associer les probabilités aux labels
    probabilities_dict = {LABELS[i]: float(probabilities[i]) for i in range(NUM_LABELS)}

    return LABELS[predicted_class], probabilities_dict


# Exemple d'utilisation
if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    audio_file = os.path.join(base_path, "colere", "c1ac.wav")

    predicted_emotion, probabilities = predict_emotion(audio_file)

    print(f"🎤 L'émotion prédite est : {predicted_emotion}")
    print(f"📊 Probabilités par classe : {probabilities}")
