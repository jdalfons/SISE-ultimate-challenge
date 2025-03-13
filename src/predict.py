import torch
import librosa
import numpy as np
from model.emotion_classifier import EmotionClassifier
from utils.preprocessing import collate_fn
from config import DEVICE, NUM_LABELS
import os

# Charger le modèle entraîné
MODEL_PATH = "acc_model.pth"
feature_dim = 40  # Nombre de MFCCs utilisés
model = EmotionClassifier(feature_dim, NUM_LABELS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Mode évaluation

# Fonction pour prédire l’émotion d’un fichier audio
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
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Définition des labels
    LABELS = {0: "colère", 1: "neutre", 2: "joie"}
    return LABELS[predicted_class]

    

#Exemple d'utilisation
if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    audio_file = os.path.join(base_path, "colere", "c1ac.wav")
    emotion = predict_emotion(audio_file)
    print(f"🎤 L'émotion prédite est : {emotion}")
