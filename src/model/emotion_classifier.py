

# Prédit 33% environ partout (dans le cas 3 classes)

# class EmotionClassifier(nn.Module):
#     def __init__(self, feature_dim, num_labels):
#         super(EmotionClassifier, self).__init__()
#         self.fc1 = nn.Linear(feature_dim, 256)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(256, num_labels)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         return self.fc2(x)
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Mécanisme d’attention permettant de pondérer l’importance des caractéristiques audio"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, sequence_length, hidden_dim)
        attention_scores = self.attention_weights(lstm_output)  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalisation softmax
        weighted_output = lstm_output * attention_weights  # Pondération des features
        return weighted_output.sum(dim=1)  # Somme pondérée sur la séquence

class EmotionClassifier(nn.Module):
    """Modèle de classification des émotions basé sur BiLSTM et attention"""
    def __init__(self, feature_dim, num_labels, hidden_dim=128):
        super(EmotionClassifier, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)  # Bidirectionnel → hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_labels)  # Couche de classification finale

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_dim*2)
        attention_out = self.attention(lstm_out)  # (batch_size, hidden_dim*2)
        logits = self.fc(attention_out)  # (batch_size, num_labels)
        return logits




