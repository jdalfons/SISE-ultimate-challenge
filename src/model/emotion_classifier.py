import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2EmotionClassifier(nn.Module):

    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53-french", num_labels=3):
        super(Wav2Vec2EmotionClassifier, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.fc = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values).last_hidden_state
        pooled_output = torch.mean(outputs, dim=1)  
        logits = self.fc(pooled_output)
        return self.softmax(logits)
