################################
### NOT YET TESTED
###############################

import streamlit as st
import pyaudio
import wave
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import numpy as np
import time

# Charger le modèle Wav2Vec2 pour la classification des émotions
model_name = "superb/wav2vec2-base-superb-er"  # Exemple de modèle pour la reconnaissance des émotions
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Paramètres audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Fonction pour prédire l'émotion à partir d'un segment audio
def predict_emotion(audio_data):
    inputs = processor(audio_data, sampling_rate=RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion = model.config.id2label[predicted_id]
    return emotion

# Interface Streamlit
st.title("Détection des émotions en temps réel")

# Boutons pour démarrer et arrêter l'enregistrement
start_button = st.button("Démarrer l'enregistrement")
stop_button = st.button("Arrêter l'enregistrement")

# Zone de visualisation des émotions en temps réel
emotion_placeholder = st.empty()
final_emotion_placeholder = st.empty()

if start_button:
    st.write("Enregistrement en cours...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    real_time_emotions = []

    while not stop_button:
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Traitement en temps réel (par tranche de 1 seconde)
        if len(frames) >= RATE // CHUNK:
            audio_segment = np.frombuffer(b''.join(frames[-(RATE // CHUNK):]), dtype=np.int16)
            emotion = predict_emotion(audio_segment)
            real_time_emotions.append(emotion)
            emotion_placeholder.line_chart(real_time_emotions)  # Affichage graphique des émotions

    # Arrêt de l'enregistrement
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Sauvegarde de l'audio enregistré
    wf = wave.open("output.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    # Prédiction finale sur tout l'audio enregistré
    full_audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    final_emotion = predict_emotion(full_audio_data)
    
    final_emotion_placeholder.write(f"Émotion finale prédite : {final_emotion}")
