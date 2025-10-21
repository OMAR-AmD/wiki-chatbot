# download_models_local.py
from sentence_transformers import SentenceTransformer
import os

# Créer le dossier cache local
os.makedirs('./model_cache', exist_ok=True)

print("Downloading embedding model to local cache...")
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./model_cache')
print("✅ Model downloaded to ./model_cache/")