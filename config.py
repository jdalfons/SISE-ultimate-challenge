import os
import torch
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("Le token Hugging Face n'a pas été trouvé dans .env")

# Labels d'émotions
LABELS = {"colere": 0, "neutre": 1, "joie": 2}
#LABELS = ["colere", "neutre", "joie"]
NUM_LABELS = len(LABELS)

# Choisir le device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Modèle Wav2Vec2
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53-french"

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
BEST_MODEL_NAME = os.path.join(BASE_DIR, "model","fr-speech-emotion-model.pth")  # Monte d'un niveau pour aller à la racine

