import soundfile as sf
import torch
import torchaudio
import numpy as np
from src.model.feature_extractor import processor # type: ignore
from src.config import DEVICE

# Resampler
resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

def preprocess_audio(batch):
    speech, sample_rate = sf.read(batch["path"], dtype="float32")

    if sample_rate != 16000:
        speech = torch.tensor(speech).unsqueeze(0)
        speech = resampler(speech).squeeze(0).numpy()
    
    batch["speech"] = speech.tolist()
    batch["sampling_rate"] = 16000
    return batch

def prepare_features(batch, max_length):
    features = processor(
        batch["speech"], 
        sampling_rate=16000,  
        padding=True,
        truncation=True,
        max_length=max_length,  
        return_tensors="pt"
    )
    batch["input_values"] = features.input_values.squeeze(0)
    batch["label"] = torch.tensor(batch["label"], dtype=torch.long)
    return batch
