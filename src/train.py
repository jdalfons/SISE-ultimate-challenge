import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from utils.dataset import load_audio_data
from utils.preprocessing import preprocess_audio, prepare_features, collate_fn
from model.emotion_classifier import EmotionClassifier
from config import DEVICE, NUM_LABELS
import os

# 🔹 Charger les données
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
ds = load_audio_data(data_dir)

# 🔹 Prétraitement des données
ds = ds.map(preprocess_audio)
ds = ds.map(lambda batch: prepare_features(batch, max_length=128))

# 🔹 Séparation en train et test
ds = ds.train_test_split(test_size=0.2)
train_ds, test_ds = ds["train"], ds["test"]

# 🔹 Création des DataLoaders avec `collate_fn`
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

# 🔹 Instancier le modèle
feature_dim = 40  # Nombre de MFCCs
classifier = EmotionClassifier(feature_dim, NUM_LABELS).to(DEVICE)

# 🔹 Fonction d'entraînement
def train_classifier(classifier, train_loader, test_loader, epochs=20, lr=2e-4):
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = 0.0

    for epoch in range(epochs):
        classifier.train()
        total_loss, correct = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            logits = classifier(inputs)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)

        # 🔹 Sauvegarde du meilleur modèle
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(classifier.state_dict(), "acc_model.pth")
            print(f"✅ Nouveau meilleur modèle sauvegardé ! Accuracy: {best_accuracy:.4f}")

        print(f"📊 Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Accuracy: {train_acc:.4f}")

    return classifier

# 🔹 Fonction d'évaluation
def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

# 🔹 Lancer l'entraînement
trained_classifier = train_classifier(classifier, train_loader, test_loader, epochs=20, lr=2e-4)

print("✅ Entraînement terminé, le meilleur modèle a été sauvegardé !")
