from sentence_transformers import SentenceTransformer

print("Downloading embedding models...")

# Modèle 1 : Bon pour texte en anglais
print("\n1. Loading: all-MiniLM-L6-v2")
model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("   ✅ Downloaded (22 MB)")

# Modèle 2 : Bon pour multilangue (français + anglais)
print("\n2. Loading: distiluse-base-multilingual-cased-v2")
model2 = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
print("   ✅ Downloaded (135 MB)")

print("\n✅ All embedding models downloaded!")
print("Models saved in: ~/.cache/huggingface/hub/")