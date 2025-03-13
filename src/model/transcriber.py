import torch
from transformers import Wav2Vec2Processor
from src.model.emotion_classifier import Wav2Vec2EmotionClassifier
import librosa

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# model = Wav2Vec2EmotionClassifier()
# model.load_state_dict(torch.load("wav2vec2_emotion.pth"))
# model.to(device)


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