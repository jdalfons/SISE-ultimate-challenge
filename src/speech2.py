import os
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf  
import torchaudio
import numpy as np
from datasets import Dataset
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor
)
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

# Charger .env pour Hugging Face API Key
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("Le token Hugging Face n'a pas été trouvé dans .env")

# Définition des labels pour la classification des émotions
LABELS = {"colere": 0, "neutre": 1, "joie": 2}
NUM_LABELS = len(LABELS)

# Charger le processeur et le modèle pour l'extraction de features
model_name = "facebook/wav2vec2-large-xlsr-53-french"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(model_name)
feature_extractor = Wav2Vec2Model.from_pretrained(model_name).to(device)

# Resampleur pour convertir en 16 kHz
resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

# Définition du classifieur amélioré
class EmotionClassifier(nn.Module):
    def __init__(self, feature_dim, num_labels):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# Instancier le classifieur
classifier = EmotionClassifier(feature_extractor.config.hidden_size, NUM_LABELS).to(device)

# Charger les fichiers audio et leurs labels
def load_audio_data(data_dir):
    data = []
    for label_name, label_id in LABELS.items():
        label_dir = os.path.join(data_dir, label_name)
        for file in os.listdir(label_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(label_dir, file)
                data.append({"path": file_path, "label": label_id})
    return Dataset.from_list(data)

# Chargement du dataset
data_dir = "./dataset"
ds = load_audio_data(data_dir)

# Charger les fichiers audio avec SoundFile et rééchantillonner à 16 kHz
def preprocess_audio(batch):
    speech, sample_rate = sf.read(batch["path"], dtype="float32")

    if sample_rate != 16000:
        speech = torch.tensor(speech).unsqueeze(0)
        speech = resampler(speech).squeeze(0).numpy()
    
    batch["speech"] = speech.tolist()  # Convertir en liste pour éviter les erreurs de PyArrow
    batch["sampling_rate"] = 16000
    return batch

ds = ds.map(preprocess_audio)

# Vérifier la distribution des longueurs des fichiers audio
lengths = [len(sample["speech"]) for sample in ds]
max_length = int(np.percentile(lengths, 95))

# Transformer l'audio en features utilisables par le modèle
def prepare_features(batch):
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

ds = ds.map(prepare_features)

# Diviser les données en train et test
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

# Fonction d'évaluation sur les données de test
def evaluate(classifier, feature_extractor, test_ds):
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_ds:
            input_values = processor(
                batch["speech"], 
                sampling_rate=16000,  
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length  
            ).input_values.to(device)

            features = feature_extractor(input_values).last_hidden_state.mean(dim=1)
            logits = classifier(features)
            predictions = logits.argmax(dim=-1)
            labels = torch.tensor(batch["label"], dtype=torch.long, device=device)

            correct += (predictions == labels).sum().item()
            total += 1

    return correct / total

# Fonction d'entraînement
def train_classifier(feature_extractor, classifier, train_ds, test_ds, epochs=10, batch_size=16):
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0.0  # Variable pour stocker la meilleure accuracy

    for epoch in range(epochs):
        classifier.train()
        total_loss, correct = 0, 0
        batch_count = 0

        for i in range(0, len(train_ds), batch_size):
            batch = train_ds[i: i + batch_size]
            optimizer.zero_grad()

            input_values = processor(
                batch["speech"], 
                sampling_rate=16000,  
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length  
            ).input_values.to(device)

            with torch.no_grad():
                features = feature_extractor(input_values).last_hidden_state.mean(dim=1)
                features = (features - features.mean()) / features.std()  # Normalisation

            logits = classifier(features)
            labels = torch.tensor(batch["label"], dtype=torch.long, device=device)

            if labels.numel() == 0:
                continue

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            batch_count += 1

        train_acc = correct / len(train_ds)
        test_acc = evaluate(classifier, feature_extractor, test_ds)
        scheduler.step()

        # Sauvegarde uniquement si l'accuracy sur test est la meilleure obtenue
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                "classifier_state_dict": classifier.state_dict(),
                "feature_extractor_state_dict": feature_extractor.state_dict(),
                "processor": processor
            }, "best_emotion_model.pth")
            print(f"✅ Nouveau meilleur modèle sauvegardé ! Accuracy Test: {best_accuracy:.4f}")

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/batch_count:.4f} - Train Accuracy: {train_acc:.4f} - Test Accuracy: {test_acc:.4f}")

    return classifier

# Entraînement
trained_classifier = train_classifier(feature_extractor, classifier, train_ds, test_ds, epochs=10, batch_size=16)

print("✅ Entraînement terminé, le meilleur modèle a été sauvegardé !")
