import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

def create_wiki_embeddings():
    """Créer les embeddings et la collection ChromaDB"""
    
    print("="*70)
    print("WIKI EMBEDDINGS CREATOR")
    print("="*70)
    
    # 1. Charger les chunks
    print("\n1. Loading processed chunks...")
    with open('processed_wiki/chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"   ✅ Loaded {len(chunks)} chunks")
    
    # 2. Initialiser le modèle d'embedding
    print("\n2. Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("   ✅ Model loaded")
    
    # 3. Créer ChromaDB client (nouvelle API)
    print("\n3. Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_data")
    
    # Supprimer collection existante si elle existe
    try:
        client.delete_collection("wiki")
        print("   ⚠️  Deleted existing collection")
    except:
        pass
    
    # Créer nouvelle collection
    collection = client.create_collection(
        name="wiki",
        metadata={"description": "Wiki documentation embeddings"}
    )
    print("   ✅ Collection created")
    
    # 4. Générer embeddings et ajouter à ChromaDB
    print("\n4. Generating embeddings and adding to ChromaDB...")
    
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Extraire contenus
        contents = [chunk['content'] for chunk in batch]
        
        # Générer embeddings
        embeddings = model.encode(contents, show_progress_bar=False)
        
        # Préparer données pour ChromaDB
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        metadatas = [
            {
                'source': chunk['source'],
                'title': chunk['title'],
                'category': chunk['category'],
                'chunk_id': chunk['chunk_id']
            }
            for chunk in batch
        ]
        
        # Ajouter à collection
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=metadatas
        )
        
        print(f"   ✅ Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    
    # 5. Vérifier
    print("\n5. Verification...")
    count = collection.count()
    print(f"   ✅ Total documents in collection: {count}")
    
    print("\n" + "="*70)
    print("✅ EMBEDDINGS CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nCollection: wiki")
    print(f"Documents: {count}")
    print(f"Storage: ./chroma_data/")
    print(f"\nNext step: Run 'python hybrid_rag_retriever.py' to test RAG")


if __name__ == '__main__':
    create_wiki_embeddings()