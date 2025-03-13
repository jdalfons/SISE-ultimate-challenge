import torch
from transformers import Wav2Vec2Processor
from model import Wav2Vec2EmotionClassifier
import librosa

# Charger le modèle et le processeur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2EmotionClassifier()
model.load_state_dict(torch.load("wav2vec2_emotion.pth"))
model.to(device)
model.eval()

emotion_labels = ["joie", "colère", "neutre"]

def predict_emotion(audio_path, output_probs=False, sampling_rate=16000):
    waveform, _ = librosa.load(audio_path, sr=sampling_rate)
    input_values = processor(waveform, return_tensors="pt", sampling_rate=sampling_rate).input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    if output_probs:
        # Appliquer softmax pour obtenir des probabilités
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convertir en numpy array et prendre le premier (et seul) élément
        probabilities = probabilities[0].detach().cpu().numpy()
        
        # Créer un dictionnaire associant chaque émotion à sa probabilité
        emotion_probabilities = {emotion: prob for emotion, prob in zip(emotion_labels, probabilities)}
        return emotion_probabilities
    else:
        # Obtenir l'émotion la plus probable (i.e. la prédiction)
        predicted_label = torch.argmax(outputs, dim=1).item()
        return emotion_labels[predicted_label]


# Exemple d'utilisation
audio_test = "data/n1ac.wav"
emotion = predict_emotion(audio_test)
print(f"Émotion détectée : {emotion}")