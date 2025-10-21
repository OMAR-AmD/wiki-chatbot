import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

class WikiRAGRetriever:
    """Classe pour faire RAG sur le wiki"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialiser le retriever"""
        print("Initializing WikiRAGRetriever...")
        
        # Load vector store with new Chroma API
        self.client = chromadb.PersistentClient(path="./chroma_data")
        self.collection = self.client.get_collection("wiki_documents")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        print("✅ WikiRAGRetriever initialized")
    
    def retrieve(self, query: str, top_k: int = 3, min_relevance: float = 0.2) -> List[Dict]:
        """
        Recherche documents pertinents
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à retourner
            min_relevance: Score minimum de pertinence
        
        Returns:
            Liste de documents pertinents
        """
        # Générer embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Rechercher dans Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2,  # Chercher plus pour filtrer
            include=["documents", "metadatas", "distances"]
        )
        
        # Formater résultats
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            similarity = 1 - (distance / 2)
            
            # Filtrer par min_relevance
            if similarity >= min_relevance:
                retrieved_docs.append({
                    'content': doc,
                    'source': metadata.get('source', 'N/A'),
                    'title': metadata.get('title', 'N/A'),
                    'category': metadata.get('category', 'N/A'),
                    'relevance': similarity
                })
        
        return retrieved_docs[:top_k]
    
    def generate_answer(self, query: str, context: List[Dict], model: str = 'llama2') -> str:
        """
        Génère une réponse basée sur les documents retrievés
        
        Args:
            query: Question de l'utilisateur
            context: Documents retrievés
            model: Modèle LLM à utiliser
        
        Returns:
            Réponse du modèle
        """
        # Formater le contexte
        context_text = "\n\n".join([
            f"[{doc['title']}]\n{doc['content']}"
            for doc in context
        ])
        
        # Créer prompt
        prompt = f"""You are a helpful wiki assistant. Answer the user's question based on the provided wiki documentation.

WIKI DOCUMENTATION:
{context_text}

USER QUESTION: {query}

Answer concisely and accurately based on the documentation. If the answer is not in the documentation, say so."""
        
        # Appeler LLM
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        
        return response['message']['content']
    
    def query(self, query: str, top_k: int = 3, model: str = 'llama2') -> Dict:
        """
        Pipeline complet : retrieve + generate
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à utiliser
            model: Modèle LLM
        
        Returns:
            Dictionnaire avec réponse et métadonnées
        """
        # Retrieve
        docs = self.retrieve(query, top_k=top_k)
        
        if not docs:
            return {
                'answer': "Sorry, I couldn't find relevant information in the wiki.",
                'sources': [],
                'success': False
            }
        
        # Generate
        answer = self.generate_answer(query, docs, model=model)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'title': doc['title'],
                    'source': doc['source'],
                    'relevance': f"{doc['relevance']:.0%}"
                }
                for doc in docs
            ],
            'success': True
        }


# Test
if __name__ == '__main__':
    retriever = WikiRAGRetriever()
    
    print("\n" + "="*70)
    print("RAG PIPELINE TEST")
    print("="*70)
    
    test_queries = [
        "How do I setup the project?",
        "What is our architecture?",
        "How do I fix a database error?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*70)
        
        result = retriever.query(query)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source['title']} ({source['relevance']}) from {source['source']}")