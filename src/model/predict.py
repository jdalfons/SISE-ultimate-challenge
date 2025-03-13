import os
import torch
from transformers import Wav2Vec2Processor
from src.model.emotion_classifier import Wav2Vec2EmotionClassifier
import librosa
import streamlit as st

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Charger le modèle et le processeur
if st.session_state.model_loaded is None:
    st.session_state.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
    st.session_state.model = Wav2Vec2EmotionClassifier()
    st.session_state.model.load_state_dict(torch.load(os.path.join("src","model","wav2vec2_emotion.pth"), map_location=torch.device('cpu')), strict=False)
    st.session_state.model_loaded = True

if st.session_state.model_loaded:
    processor = st.session_state.processor
    model = st.session_state.model
    model.to(device)
    model.eval()

emotion_labels = ["joie", "colère", "neutre"]

def predict_emotion(audio_path, output_probs=False, sampling_rate=16000):
    # waveform, _ = librosa.load(audio_path, sr=sampling_rate)
    input_values = processor(audio_path, return_tensors="pt", sampling_rate=sampling_rate).input_values
    input_values = input_values.to(device)
    
    with torch.no_grad():
        outputs = model(input_values)
    
    if output_probs:        
        # Appliquer softmax pour obtenir des probabilités
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        
        # Convertir en numpy array et prendre le premier (et seul) élément
        probabilities = probabilities[0].detach().cpu().numpy()
        
        # Créer un dictionnaire associant chaque émotion à sa probabilité
        emotion_probabilities = {emotion: prob for emotion, prob in zip(emotion_labels, probabilities)}
        # emotion_probabilities = {"emotions": [emotion for emotion in emotion_labels],
        #                          "probabilities": [prob for prob in probabilities]}
        return emotion_probabilities
    else:
        # Obtenir l'émotion la plus probable (i.e. la prédiction)
        predicted_label = torch.argmax(outputs, dim=1).item()
        return emotion_labels[predicted_label]


# Exemple d'utilisation
# audio_test = "data/n1ac.wav"
# emotion = predict_emotion(audio_test)
# print(f"Émotion détectée : {emotion}")