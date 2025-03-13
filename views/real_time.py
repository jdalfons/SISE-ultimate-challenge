################################
### Real time prediction for real time record
###############################

import streamlit as st
import pyaudio
import wave
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import matplotlib.pyplot as plt
import numpy as np
import time
from predict import predict_emotion

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
# def predict_emotion(audio_data):
#     inputs = processor(audio_data, sampling_rate=RATE, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     predicted_id = torch.argmax(logits, dim=-1).item()
#     emotion = model.config.id2label[predicted_id]
#     return emotion

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
            emotion = predict_emotion(audio_segment, output_probs=False, sampling_rate=RATE)
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


################################
### Real time prediction for uploaded audio file
###############################
# Charger le modèle wav2vec et le processeur

# # Configuration Streamlit
# st.title("Analyse des émotions en temps réel")
# uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["wav", "mp3"])

# if uploaded_file is not None:
#     # Charger et rééchantillonner l'audio
#     audio, sr = librosa.load(uploaded_file, sr=16000)
    
#     # Paramètres de la fenêtre glissante
#     window_size = 1  # en secondes
#     hop_length = 0.5  # en secondes
    
#     # Créer un graphique en temps réel
#     fig, ax = plt.subplots()
#     lines = [ax.plot([], [], label=emotion)[0] for emotion in emotions]
#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, len(audio) / sr)
#     ax.set_xlabel("Temps (s)")
#     ax.set_ylabel("Probabilité")
#     ax.legend()
    
#     chart = st.pyplot(fig)
    
#     # Traitement par fenêtre glissante
#     for i in range(0, len(audio), int(hop_length * sr)):
#         chunk = audio[i:i + int(window_size * sr)]
#         if len(chunk) < int(window_size * sr):
#             break
        
#         emotion_scores = predict_emotion(chunk, output_probs=False, sampling_rate=RATE)
        
#         # Mettre à jour le graphique
#         for emotion, line in zip(emotions, lines):
#             xdata = line.get_xdata().tolist()
#             ydata = line.get_ydata().tolist()
#             xdata.append(i / sr)
#             ydata.append(emotion_scores[emotion])
#             line.set_data(xdata, ydata)
        
#         ax.relim()
#         ax.autoscale_view()
#         chart.pyplot(fig)
        
#     st.success("Analyse terminée !")







############################################
### Progress bar
############################################

with st.status("Downloading data...", expanded=True) as status:
    st.write("Searching for data...")
    time.sleep(2)
    st.write("Found URL.")
    time.sleep(1)
    st.write("Downloading data...")
    time.sleep(1)
    status.update(
        label="Download complete!", state="complete", expanded=False
    )

st.button("Rerun")


############################################
### Time duration estimation
############################################
progress_bar = st.progress(0)
time_placeholder = st.empty()

total_time = 10  # Total estimated time in seconds
for i in range(total_time):
    # Update progress bar
    progress_bar.progress((i + 1) / total_time)
    
    # Update time estimation
    remaining_time = total_time - i - 1
    time_placeholder.text(f"Estimated time remaining: {remaining_time} seconds")
    
    # Simulate task progress
    time.sleep(1)



############################################
### Audio file noise reduction
############################################
from pydub import AudioSegment
import noisereduce as nr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Fonction de réduction de bruit
def reduce_noise(audio_data, sr):
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr)
    return reduced_noise

# Chargement du modèle wav2vec
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Interface Streamlit
st.title("Application de transcription audio avec réduction de bruit")

uploaded_file = st.file_uploader("Choisissez un fichier audio .wav", type="wav")

if uploaded_file is not None:
    # Chargement et prétraitement de l'audio
    audio = AudioSegment.from_wav(uploaded_file)
    audio_array = np.array(audio.get_array_of_samples())
    
    # Réduction de bruit
    reduced_noise_audio = reduce_noise(audio_array, audio.frame_rate)
    
    # Traitement avec wav2vec
    input_values = processor(reduced_noise_audio, sampling_rate=audio.frame_rate, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    st.audio(uploaded_file, format="audio/wav")
    st.write("Transcription:")
    st.write(transcription)


############################################
### Choix des émotions
############################################
# options = ['Sadness','Anger', 'Disgust', 'Fear', 'Surprise', 'Joy','Neutral']
# selected_options = st.multiselect('What emotions do you want to be displayed', options, default=['Joy', 'Anger','Neutral])


############################################
### Transcription Speech2Text
############################################
# # Fonction pour transcrire l'audio
# def transcribe_audio(audio):
#     # Préparer les données d'entrée pour le modèle
#     input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    
#     # Passer les données dans le modèle pour obtenir les logits
#     with torch.no_grad():
#         logits = model(input_values).logits
    
#     # Décoder les prédictions en texte
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)[0]
#     return transcription

# # Charger et transcrire l'audio
# # audio, rate = load_audio(audio_file_path) # (re)chargement de l'audio si nécessaire
# transcription = transcribe_audio(audio)

# # Afficher la transcription
# print("Transcription :", transcription)


############################################
### Feedback
############################################
import pandas as pd
import os

# Initialisation du fichier CSV
csv_file = "predictions/feedback.csv"

# Vérifier si le fichier CSV existe, sinon le créer avec des colonnes appropriées
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["filepath", "prediction", "feedback"])
    df.to_csv(csv_file, index=False)

# Charger les données existantes du CSV
df = pd.read_csv(csv_file)

# Interface Streamlit
st.title("Predicted emotion feedback")

# Simuler une prédiction pour l'exemple (remplacez par votre modèle réel)
audio_file_name = "example_audio.wav"
predicted_emotion = "Joie"  # Exemple de prédiction

st.write(f"Fichier audio : {audio_file_name}")
st.write(f"Émotion détectée : {predicted_emotion}")

# Formulaire de feedback
with st.form("feedback_form"):
    st.write("Est-ce la bonne émotion qui a été détectée ? Cochez la réelle émotion.")
    feedback = st.selectbox("Votre réponse :", ['Sadness','Anger', 'Disgust', 'Fear', 'Surprise', 'Joy', 'Neutral'])
    submit_button = st.form_submit_button("Soumettre")
    st.write("En cliquant sur ce bouton, vous acceptez que votre audio soit sauvegardé dans notre base de données.")

    if submit_button:
        # Ajouter le feedback au DataFrame
        new_entry = {"filepath": audio_file_name, "prediction": predicted_emotion, "feedback": feedback}
        df = df.append(new_entry, ignore_index=True)
        
        # Sauvegarder les données mises à jour dans le fichier CSV
        df.to_csv(csv_file, index=False)

        # Sauvegarder le fichier audio
        with open("predictions/data", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Confirmation pour l'utilisateur
        st.success("Merci pour votre retour ! Vos données ont été sauvegardées.")

# Afficher les données sauvegardées (optionnel)
# st.write("Données collectées jusqu'à présent :")
# st.dataframe(df)















############################################
### Predict proba (to replace in predict.py)
############################################
import librosa
def predict_emotion_probabilities(audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000)
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
    
    # Appliquer softmax pour obtenir des probabilités
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Convertir en numpy array et prendre le premier (et seul) élément
    probabilities = probabilities[0].detach().cpu().numpy()
    
    # Créer un dictionnaire associant chaque émotion à sa probabilité
    emotion_probabilities = {emotion: prob for emotion, prob in zip(emotion_labels, probabilities)}
    
    return emotion_probabilities