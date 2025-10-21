# download_models.py
from sentence_transformers import SentenceTransformer

print("Downloading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model downloaded and cached!")