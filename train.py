import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from utils.dataset import load_audio_data
from utils.preprocessing import preprocess_audio, prepare_features, collate_fn
from model.emotion_classifier import EmotionClassifier
from config import DEVICE, NUM_LABELS, BEST_MODEL_NAME
import os

# Charger les données et les séparer en train / test
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
ds = load_audio_data(data_dir)

# Prétraitement
ds["train"] = ds["train"].map(preprocess_audio).map(lambda batch: prepare_features(batch, max_length=128))
ds["test"] = ds["test"].map(preprocess_audio).map(lambda batch: prepare_features(batch, max_length=128))

# DataLoader
train_loader = DataLoader(ds["train"], batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(ds["test"], batch_size=8, shuffle=False, collate_fn=collate_fn)

# Instancier le modèle
classifier = EmotionClassifier(feature_dim=40, num_labels=NUM_LABELS).to(DEVICE)

# Fonction d'entraînement
def train_classifier(classifier, train_loader, test_loader, epochs=20):
    optimizer = optim.AdamW(classifier.parameters(), lr=2e-5, weight_decay=0.01)
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

        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(classifier.state_dict(), BEST_MODEL_NAME)
            print(f"✔️ Nouveau meilleur modèle sauvegardé ! Accuracy: {best_accuracy:.4f}")

        print(f"📢 Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Accuracy: {train_acc:.4f}")

    return classifier

# Évaluer le modèle
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

# Lancer l'entraînement
trained_classifier = train_classifier(classifier, train_loader, test_loader, epochs=20)

print("✅ Entraînement terminé, le meilleur modèle a été sauvegardé !")
