import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from emotion_dataset import EmotionDataset
from emotion_classifier import Wav2Vec2EmotionClassifier
import os
from utils import collate_fn


# Charger le processeur et le dataset
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv"))
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Le fichier {data_path} est introuvable.")

dataset = EmotionDataset(data_path, processor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)  # collate_fn ajouté


# Initialiser le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2EmotionClassifier().to(device)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Entraînement du modèle
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Sauvegarde du modèle
torch.save(model.state_dict(), "wav2vec2_emotion.pth")
print("Modèle sauvegardé !")
