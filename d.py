import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
import json

class RAGPipeline:
    def __init__(self, 
                 collection_name="wiki_data",
                 embedding_model="all-MiniLM-L6-v2",
                 persist_directory="./chroma_db"):
        """
        Initialise le pipeline RAG
        
        Args:
            collection_name: Nom de la collection Chroma
            embedding_model: Modèle pour les embeddings
            persist_directory: Dossier de stockage Chroma
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialiser Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Créer ou récupérer la collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ RAG Pipeline initialized")
        print(f"   Collection: {collection_name}")
        print(f"   Documents: {self.collection.count()}")
    
    def add_documents(self, documents: List[Dict]):
        """
        Ajoute des documents à la knowledge base
        
        Args:
            documents: Liste de dicts avec 'content', 'title', 'source'
        """
        if not documents:
            print("⚠️ No documents to add")
            return
        
        # Préparer les données
        texts = [doc['content'] for doc in documents]
        metadatas = [
            {
                'title': doc.get('title', 'Unknown'),
                'source': doc.get('source', 'Unknown')
            }
            for doc in documents
        ]
        
        # Générer IDs uniques basés sur le timestamp et index
        import time
        timestamp = int(time.time() * 1000)
        ids = [f"doc_{timestamp}_{i}" for i in range(len(documents))]
        
        # Générer embeddings
        print(f"🔄 Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Ajouter à Chroma
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Added {len(documents)} documents to knowledge base")
        except Exception as e:
            print(f"⚠️ Error adding documents: {e}")
            print("   Trying to upsert instead...")
            self.collection.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Upserted {len(documents)} documents to knowledge base")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Recherche les documents pertinents
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à retourner
            
        Returns:
            Liste de documents avec leur score de pertinence
        """
        if self.collection.count() == 0:
            print("⚠️ Knowledge base is empty")
            return []
        
        # Générer embedding de la query
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Rechercher dans Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Formater les résultats
        documents = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = {
                    'content': results['documents'][0][i],
                    'title': results['metadatas'][0][i].get('title', 'Unknown'),
                    'source': results['metadatas'][0][i].get('source', 'Unknown'),
                    'score': 1 - results['distances'][0][i]  # Convertir distance en score
                }
                documents.append(doc)
        
        return documents
    
    def load_from_directory(self, directory_path: str):
        """
        Charge tous les fichiers .txt d'un dossier
        
        Args:
            directory_path: Chemin du dossier contenant les docs
        """
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return
        
        documents = []
        
        # Parcourir tous les fichiers .txt
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt') or filename.endswith('.md'):
                filepath = os.path.join(directory_path, filename)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Découper en chunks si trop long
                chunks = self._chunk_text(content, max_length=500)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'title': f"{filename} (part {i+1})",
                        'source': filename
                    })
        
        print(f"📁 Found {len(documents)} chunks from {directory_path}")
        self.add_documents(documents)
    
    def _chunk_text(self, text: str, max_length: int = 500) -> List[str]:
        """
        Découpe un texte en chunks de taille raisonnable
        
        Args:
            text: Texte à découper
            max_length: Taille max d'un chunk (en mots)
            
        Returns:
            Liste de chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length):
            chunk = ' '.join(words[i:i + max_length])
            chunks.append(chunk)
        
        return chunks
    
    def clear_collection(self):
        """Vide la collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️ Collection cleared")


# Test rapide
if __name__ == "__main__":
    # Créer le pipeline
    rag = RAGPipeline()
    
    # Exemple: Ajouter des documents manuellement
    sample_docs = [
        {
            'content': 'The RAG pipeline uses ChromaDB for vector storage and Sentence-Transformers for embeddings.',
            'title': 'RAG Architecture',
            'source': 'architecture.md'
        },
        {
            'content': 'The agent has three main tools: generate_tests, extract_metrics, and suggest_refactor.',
            'title': 'Agent Tools',
            'source': 'agent.md'
        },
        {
            'content': 'Ollama runs locally and provides access to Llama models without internet connection.',
            'title': 'Local LLM',
            'source': 'setup.md'
        }
    ]
    
    # Ajouter les docs
    rag.add_documents(sample_docs)
    
    # Tester la recherche
    print("\n=== Test Search ===")
    results = rag.search("What are the agent tools?", top_k=2)
    
    for i, doc in enumerate(results):
        print(f"\n📄 Result {i+1}:")
        print(f"   Title: {doc['title']}")
        print(f"   Score: {doc['score']:.3f}")
        print(f"   Content: {doc['content'][:100]}...")