import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from utils.dataset import load_audio_data
from utils.preprocessing import preprocess_audio, prepare_features
from model.emotion_classifier import EmotionClassifier
from model.feature_extrator import feature_extractor, processor
from config import DEVICE, NUM_LABELS
import os

# Charger les données
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
print(f"data dir  {data_dir}")
ds = load_audio_data(data_dir)

# Prétraitement
ds = ds.map(preprocess_audio)

# Ajustement de la longueur maximale
lengths = [len(sample["speech"]) for sample in ds]
max_length = int(np.percentile(lengths, 95))

ds = ds.map(lambda batch: prepare_features(batch, max_length))

# Séparation en train et test
ds = ds.train_test_split(test_size=0.2)
train_ds, test_ds = ds["train"], ds["test"]

# Instancier le modèle
classifier = EmotionClassifier(feature_extractor.config.hidden_size, NUM_LABELS).to(DEVICE)

# Fonction d'entraînement
def train_classifier(classifier, train_ds, test_ds, epochs=20, batch_size=8):
    optimizer = optim.AdamW(classifier.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = 0.0

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
            ).input_values.to(DEVICE)

            with torch.no_grad():
                features = feature_extractor(input_values).last_hidden_state.mean(dim=1)

            logits = classifier(features)
            labels = torch.tensor(batch["label"], dtype=torch.long, device=DEVICE)

            if labels.numel() == 0:
                continue

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            batch_count += 1

        train_acc = correct / len(train_ds)

        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save({
                "classifier_state_dict": classifier.state_dict(),
                "feature_extractor_state_dict": feature_extractor.state_dict(),
                "processor": processor
            }, "acc_model.pth")
            print(f"Nouveau meilleur modèle sauvegardé ! Accuracy: {best_accuracy:.4f}")

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/batch_count:.4f} - Accuracy: {train_acc:.4f}")

    return classifier

# Lancer l'entraînement
trained_classifier = train_classifier(classifier, train_ds, test_ds, epochs=20, batch_size=8)

print("✅ Entraînement terminé, le meilleur modèle a été sauvegardé !")
