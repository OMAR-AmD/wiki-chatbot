import chromadb
from sentence_transformers import SentenceTransformer
from typing import List
import pickle
import os
import json

def load_chunks() -> List[str]:
    """Load chunks from JSON file (processed_wiki/chunks_smart.json)"""
    chunks_path = "processed_wiki/chunks_smart.json"

    if not os.path.exists(chunks_path):
        print("⚠️ Fichier JSON introuvable :", chunks_path)
        return []

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extraire la bonne clé ('content' ou 'text')
    if isinstance(data, list) and isinstance(data[0], dict) and "content" in data[0]:
        chunks = [chunk["content"] for chunk in data]
    elif isinstance(data, list) and isinstance(data[0], dict) and "text" in data[0]:
        chunks = [chunk["text"] for chunk in data]
    else:
        print("⚠️ Format inattendu du JSON (aucune clé 'content' ou 'text' trouvée).")
        return []

    # Corriger uniquement si on détecte une corruption typique (Ã©, Ã¨, etc.)
    def fix_encoding(s: str) -> str:
        if any(bad in s for bad in ["Ã", "Â", "¤"]):
            try:
                return s.encode("latin1").decode("utf-8")
            except Exception:
                return s
        return s

    chunks = [fix_encoding(c) for c in chunks]

    print(f"   ✅ Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def create_vector_store():
    print("Creating Vector Store...")
    
    # 1. Load chunks
    print("1. Loading chunks...")
    chunks = load_chunks()
    print(f"   ✅ Loaded {len(chunks)} chunks")
    
    # 2. Initialize Chroma with new API
    print("2. Initializing Chroma...")
    try:
        # New Chroma API - use PersistentClient for local storage
        client = chromadb.PersistentClient(path="./chroma_data")
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name="wiki_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("   ✅ Chroma initialized")
        
    except Exception as e:
        print(f"   ❌ Error initializing Chroma: {e}")
        raise
    
    # 3. Initialize embedding model
    print("3. Initializing embedding model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Embedding model loaded")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        raise
    
    # 4. Add chunks to vector store
    print("4. Adding chunks to vector store...")
    try:
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = model.encode(chunk, convert_to_tensor=False).tolist()
            
            # Add to collection
            collection.add(
                ids=[f"chunk_{i}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": "wiki_document"}]
            )
            
            if (i + 1) % 5 == 0:
                print(f"   📝 Processed {i + 1}/{len(chunks)} chunks")
        
        print(f"   ✅ Added all {len(chunks)} chunks to vector store")
        
    except Exception as e:
        print(f"   ❌ Error adding chunks: {e}")
        raise
    
    # 5. Persist the collection
    print("5. Persisting vector store...")
    try:
        # PersistentClient automatically saves to disk
        print("   ✅ Vector store persisted to ./chroma_data")
    except Exception as e:
        print(f"   ❌ Error persisting: {e}")
        raise
    
    print("\n✨ Vector store created successfully!")

if __name__ == "__main__":
    create_vector_store()