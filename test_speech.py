import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import os

# 🔹 Paramètres
MODEL_NAME = "./wav2vec2_emotion"  # Chemin du modèle sauvegardé
LABELS = ["colere", "joie", "neutre"]  # Les classes

# 🔹 Charger le processeur et le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()  # Mode évaluation


def predict_emotion(audio_path):
    # Charger l'audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Prétraitement du son
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32000  # Ajuste selon la durée de tes fichiers
    )

    # Envoyer les données sur le bon device (CPU ou GPU)
    input_values = inputs["input_values"].to(device)

    # Prédiction
    with torch.no_grad():
        logits = model(input_values).logits

    # Trouver l'émotion prédite
    predicted_class = torch.argmax(logits, dim=-1).item()

    return LABELS[predicted_class]  # Retourne le label correspondant

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
audio_file = os.path.join(base_path, "colere", "c1ac.wav")
predicted_emotion = predict_emotion(audio_file)
print(f"🎙️ Émotion prédite : {predicted_emotion}")



