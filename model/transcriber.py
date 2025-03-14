import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Charger le modèle et le processeur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-french"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
model.eval()

def transcribe_audio(audio_path, sampling_rate=16000):
    # Charger l'audio
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    # Transformer l'audio en entrée pour le modèle
    input_values = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values.to(device)

    # Obtenir les prédictions
    with torch.no_grad():
        logits = model(input_values).logits

    # Décoder les prédictions en texte
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Exemple d'utilisation
if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    audio_path = os.path.join(base_path, "colere", "c1af.wav")
    texte = transcribe_audio(audio_path)
    print(f"Transcription : {texte}")
