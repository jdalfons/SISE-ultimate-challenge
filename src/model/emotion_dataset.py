import librosa
import torch
import pandas as pd
from torch.utils.data import Dataset
import os

class EmotionDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file, sep=",", header=0)
        # print(self.data.info())  # Pour voir les premières lignes du dataset
        self.processor = processor
        self.emotion_labels = {"joie": 0, "colere": 1, "neutre": 2}
        # print(self.data["emotion"].unique())  # Pour voir les valeurs exactes


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        audio_file = self.data.iloc[idx, 0]
        label = self.emotion_labels[self.data.iloc[idx, 1].strip()]

        audio_path = os.path.join(base_path, audio_file)
        waveform, _ = librosa.load(audio_path, sr=16000)  # Chargement audio
        input_values = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

        return input_values.squeeze(0), torch.tensor(label, dtype=torch.long)

