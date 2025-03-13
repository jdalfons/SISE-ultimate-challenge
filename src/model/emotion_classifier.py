import torch
import torch.nn as nn

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
    


class EmotionClassifier(nn.Module):
    def __init__(self, feature_dim, num_labels=3):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_labels)
        self.dropout = nn.Dropout(0.3)  # Evite l'overfitting

    def forward(self, x):
        pooled_output = torch.mean(x, dim=1)  # Moyenne des features audio
        pooled_output = self.dropout(pooled_output)  # Dropout avant classification
        logits = self.fc(pooled_output)
        return logits  



