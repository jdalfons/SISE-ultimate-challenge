from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Charger le modèle et le processeur Wav2Vec 2.0
model_name = "facebook/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Charger l'audio
audio_file = "C:\\Users\\fkpamegan\\Downloads\\datasets_oreau2_m_sessp_07a01Pa.wav"
y, sr = librosa.load(audio_file, sr=16000) 

# Prétraiter l'audio avec le processeur Wav2Vec 2.0
input_values = processor(y, return_tensors="pt").input_values

# Obtenir la prédiction (logits)
with torch.no_grad():
    logits = model(input_values).logitsa

# Obtenir les IDs des tokens prédits (transcription)
predicted_ids = torch.argmax(logits, dim=-1)

# Décoder les IDs pour obtenir le texte transcrit
transcription = processor.decode(predicted_ids[0])

print("Transcription:", transcription)


# Extraire le pitch (hauteur tonale) et l'intensité
pitch, magnitudes = librosa.core.piptrack(y=y, sr=sr)
intensity = librosa.feature.rms(y=y)  # Intensité (volume)

# Calculer le tempo (vitesse de parole)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

# Affichage du pitch
plt.figure(figsize=(10, 6))
librosa.display.specshow(pitch, x_axis='time', y_axis='log')
plt.colorbar()
plt.title("Pitch (Hauteur Tonale)")
plt.show()

# Affichage de l'intensité
plt.figure(figsize=(10, 6))
librosa.display.specshow(intensity, x_axis='time')
plt.colorbar()
plt.title("Intensité")
plt.show()

# Fusionner la transcription avec les caractéristiques prosodiques (pitch, intensité, tempo)
features = np.hstack([
    np.mean(intensity, axis=1),  # Moyenne de l'intensité
    np.mean(pitch, axis=1),  # Moyenne du pitch
    tempo  # Tempo
])

# Afficher les caractéristiques extraites
print("Caractéristiques combinées :")
print(features)
