import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from src.model.emotion_classifier import EmotionClassifier
from src.model.feature_extractor import feature_extractor, processor
from src.utils.preprocessing import resampler
from src.config import DEVICE, LABELS
import os


# Charger le modèle sauvegardé
classifier = EmotionClassifier(feature_extractor.config.hidden_size, len(LABELS)).to(DEVICE)
classifier.load_state_dict(torch.load(os.path.join("src","model","best_emotion_model.pth"), map_location=torch.device(DEVICE)), strict=False)
classifier.eval()


# Fonction de prédiction
def predict_emotion(speech, output_probs=False, sampling_rate=16000):
    # Charger l'audio
    # waveform, sample_rate = librosa.load(speech, sr=None)
    # speech_audio, sample_rate = sf.read(speech, dtype="float32")

    # Rééchantillonnage si nécessaire
    # if sample_rate != sampling_rate:
    #     speech = torch.tensor(speech).unsqueeze(0)
    #     speech = resampler(speech).squeeze(0).numpy()

    # Extraire les features
    inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        features = feature_extractor(input_values).last_hidden_state.mean(dim=1)
        logits = classifier(features)

    if output_probs:        
        # Appliquer softmax pour obtenir des probabilités
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Convertir en numpy array et prendre le premier (et seul) élément
        probabilities = probabilities[0].detach().cpu().numpy()
        
        # Créer un dictionnaire associant chaque émotion à sa probabilité
        emotion_probabilities = {emotion: prob for emotion, prob in zip(LABELS, probabilities)}
        # emotion_probabilities = {"emotions": [emotion for emotion in emotion_labels],
        #                          "probabilities": [prob for prob in probabilities]}
        return emotion_probabilities
    else:
        # Obtenir l'émotion la plus probable (i.e. la prédiction)
        predicted_label = torch.argmax(logits, dim=-1).item()
        emotion = LABELS[predicted_label]

        return emotion
    

# Exemple d'utilisation
# if __name__ == "__main__":
#     base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
#     audio_file = os.path.join(base_path, "colere", "c1ac.wav")
#     emotion = predict_emotion(audio_file)
#     print(f"🎤 L'émotion prédite est : {emotion}")
