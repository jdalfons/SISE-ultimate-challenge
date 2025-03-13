import torch
import torchaudio
import os
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer

# 🔹 Paramètres
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-french"
NUM_LABELS = 3  # Nombre de classes émotionnelles
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_LENGTH = 32000  # Ajuste selon la durée de tes fichiers audio

# 🔹 Vérifier GPU dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Charger le processeur et le modèle
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="single_label_classification"
).to(device)

# 🔹 Fonction pour charger les fichiers audio sans CSV
def load_audio_data(data_dir):
    data = {"file_path": [], "label": []}
    labels = ["colere", "joie", "neutre"]  # Ajuste selon tes classes

    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                data["file_path"].append(os.path.join(folder_path, file))
                data["label"].append(labels.index(label))

    dataset = Dataset.from_dict(data)
    train_test_split = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test
    return DatasetDict({"train": train_test_split["train"], "test": train_test_split["test"]})

# 🔹 Prétraitement de l'audio
def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH  # ✅ Correction de l'erreur
    )
    return inputs["input_values"][0]  # Récupère les valeurs audio prétraitées

# 🔹 Charger et prétraiter le dataset
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
ds = load_audio_data(data_dir)

def preprocess_batch(batch):
    batch["input_values"] = preprocess_audio(batch["file_path"])
    return batch

ds = ds.map(preprocess_batch, remove_columns=["file_path"])

# 🔹 Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./wav2vec2_emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# 🔹 Définir le trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)

# 🚀 Lancer l'entraînement
trainer.train()
