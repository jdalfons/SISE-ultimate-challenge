import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, labels = zip(*batch)  # Séparer les features et les labels
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # Padding des audios
    labels = torch.tensor(labels, dtype=torch.long)  # Conversion en tensor
    return inputs, labels
