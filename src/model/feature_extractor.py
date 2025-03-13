import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from src.config import MODEL_NAME, DEVICE

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
feature_extractor = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
