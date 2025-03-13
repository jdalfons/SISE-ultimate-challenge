import os
from datasets import Dataset
from config import LABELS
import pandas as pd

def load_audio_data(data_dir):
    data = []
    for label_name, label_id in LABELS.items():
        label_dir = os.path.join(data_dir, label_name)
        for file in os.listdir(label_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(label_dir, file)
                data.append({"path": file_path, "label": label_id})
    return Dataset.from_list(data)


# def load_audio_data_from_csv(csv_path, data_dir):
#     data = []
#     df = pd.read_csv(csv_path, sep=",", header=0)
#     print(df.head())

#     for _, row in df.iterrows():
#         file_path = os.path.join(data_dir, row["dossier"])
#         label = row["emotion"]

#         if os.path.exists(file_path) and label in LABELS:
#             data.append({"path": file_path, "label": LABELS[label]})
#         else:
#             print(f"⚠️ Fichier manquant ou label inconnu : {file_path} - {label}")

#     return Dataset.from_list(data)

# #Charger le dataset à partir du CSV
# csv_path = os.path.abspath(os.path.join(os.path.dirname(file), "new_data", "dataset.csv"))
# data_dir = os.path.abspath(os.path.join(os.path.dirname(file), "new_data"))
# ds = load_audio_data_from_csv(csv_path, data_dir)
