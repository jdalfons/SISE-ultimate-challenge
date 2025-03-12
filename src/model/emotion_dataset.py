import librosa
import torch
import pandas as pd
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.emotion_labels = {"joie": 0, "colère": 1, "neutre": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx, 0]
        label = self.emotion_labels[self.data.iloc[idx, 1]]

        waveform, _ = librosa.load(audio_path, sr=16000)
        input_values = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

        return input_values.squeeze(0), torch.tensor(label, dtype=torch.long)
