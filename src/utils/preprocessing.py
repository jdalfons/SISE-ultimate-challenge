import librosa
import soundfile as sf
import torch
import torchaudio
import numpy as np
from model.feature_extractor import processor  # type: ignore
from config import DEVICE

# Resampler pour convertir en 16kHz
resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

def preprocess_audio(batch):
    speech, sample_rate = sf.read(batch["path"], dtype="float32")

    # Convertir en numpy array si ce n'est pas déjà le cas
    speech = np.array(speech, dtype=np.float32)

    # Vérifier que le format est bien float32
    if speech.dtype != np.float32:
        raise ValueError(f"Le fichier {batch['path']} n'est pas en float32.")

    # Resampling si nécessaire
    if sample_rate != 16000:
        speech = torch.tensor(speech).unsqueeze(0)  # Ajouter une dimension pour le resampler
        speech = resampler(speech).squeeze(0).numpy()  # Appliquer le resampler et reconvertir en numpy array

    batch["speech"] = speech
    batch["sampling_rate"] = 16000
    return batch


def prepare_features(batch, max_length):
    y, sr = batch["speech"], 16000

    # S'assurer que y est bien un numpy array float32
    if not isinstance(y, np.ndarray):
        y = np.array(y, dtype=np.float32)

    # Vérifier que les valeurs sont bien en float32
    y = y.astype(np.float32)

    # Extraction des MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Debugging: afficher la forme des MFCCs
    print(f"MFCC original shape: {mfcc.shape}")

    # Ajuster la longueur des MFCCs
    if mfcc.shape[1] > max_length:
        mfcc = mfcc[:, :max_length]  # Tronquer si trop long
    else:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')  # Padding si trop court

    print(f"MFCC padded shape: {mfcc.shape}")

    # Convertir en tensor PyTorch et stocker
    batch["input_values"] = torch.tensor(mfcc.T, dtype=torch.float32)  # Transposer pour obtenir (max_length, 40)
    return batch


def collate_fn(batch):
    """ Fonction pour assembler les batchs avec padding des features """
    
    # Récupérer les features (MFCC) et les labels
    inputs = [np.array(sample["input_values"], dtype=np.float32) for sample in batch]
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)

    # Vérifier que inputs est bien une liste de numpy arrays
    #print(f"Types des inputs: {[type(x) for x in inputs]}")  # Debugging

    # Trouver la longueur max des MFCC dans ce batch
    max_length = max([x.shape[0] for x in inputs])

    # Appliquer un padding avec des zéros pour uniformiser les tailles
    padded_inputs = [np.pad(x, ((0, max_length - x.shape[0]), (0, 0)), mode="constant") for x in inputs]

    # Convertir en tenseur PyTorch
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.float32)

    return inputs_tensor, labels

