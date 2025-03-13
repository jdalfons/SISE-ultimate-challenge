import os
import torch
from transformers import Wav2Vec2Processor
from src.model.emotion_classifier import Wav2Vec2EmotionClassifier
import librosa

# Charger le modèle et le processeur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if st.
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2EmotionClassifier()
model.load_state_dict(torch.load(os.path.join("src","model","wav2vec2_emotion.pth"), map_location=torch.device('cpu')), strict=False)
model.to(device)


def transcribe_audio(audio, sampling_rate=16000):
    # Préparer les données d'entrée pour le modèle
    input_values = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values

    # Passer les données dans le modèle pour obtenir les logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Décoder les prédictions en texte
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription