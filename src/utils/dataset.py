import os
from datasets import Dataset
from config import LABELS

def load_audio_data(data_dir):
    data = []
    for label_name, label_id in LABELS.items():
        label_dir = os.path.join(data_dir, label_name)
        for file in os.listdir(label_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(label_dir, file)
                data.append({"path": file_path, "label": label_id})
    return Dataset.from_list(data)
