# Fichier: check_download.py
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

print("--- VÉRIFICATION DES MODÈLES ---")
# Téléchargement du modèle d'embedding (SentenceTransformer)
print("1. Tentative de chargement du modèle d'embedding...")
model_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"   ✅ Modèle d'embedding chargé.")

# Téléchargement du modèle de reranking (CrossEncoder)
print("2. Tentative de chargement du modèle de reranking...")
model_rerank = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print(f"   ✅ Modèle de reranking chargé.")

print("\n🎉 Les deux modèles sont maintenant dans votre cache local.")