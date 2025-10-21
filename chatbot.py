import ollama
from rag_pipeline import RAGPipeline
from datetime import datetime
import json
import os

class WikiChatbot:
    def __init__(self, model_name="llama2"):
        """
        Initialise le chatbot Wiki
        
        Args:
            model_name: Nom du modèle Ollama à utiliser (par défaut: llama2)
        """
        self.model_name = model_name
        self.rag = RAGPipeline(collection_name="wiki_data")
        self.conversation_history = []
    
    def generate_hybrid_response(self, query):
        """
        Génère une réponse intelligente en mode hybride:
        - Cherche dans la knowledge base
        - Si documents pertinents trouvés (score > 0.5): utilise RAG + LLM
        - Sinon: utilise seulement les connaissances générales du LLM
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Dict avec 'answer', 'sources', 'type'
        """
        # Chercher dans la knowledge base
        relevant_docs = self.rag.search(query, top_k=3)
        
        # Filtrer seulement les documents avec un score élevé (> 0.5)
        high_quality_docs = [doc for doc in relevant_docs if doc.get('score', 0) > 0.5]
        
        if high_quality_docs and self.rag.collection.count() > 0:
            # Mode RAG: Utiliser la documentation + connaissances du LLM
            context = "\n\n".join([
                f"Document {i+1}:\n{doc['content']}" 
                for i, doc in enumerate(high_quality_docs)
            ])
            
            prompt = f"""You are a helpful AI assistant with access to project documentation.

Relevant documentation found:
{context}

User question: {query}

Instructions:
- Answer the question using BOTH your general knowledge AND the documentation provided
- If the documentation is relevant, cite it in your answer
- If the documentation doesn't fully answer the question, supplement with your general knowledge
- Be natural and conversational
- Keep responses concise but informative

Answer:"""
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            # Préparer les sources
            sources = [
                {
                    'title': doc.get('title', 'Unknown'),
                    'source': doc.get('source', 'Unknown'),
                    'relevance': f"{doc.get('score', 0):.2f}"
                }
                for doc in high_quality_docs
            ]
            
            return {
                'answer': response['response'].strip(),
                'sources': sources,
                'type': 'hybrid_rag'
            }
        
        else:
            # Mode Normal: LLM utilise ses connaissances générales
            prompt = f"""You are a helpful and knowledgeable AI assistant.

User question: {query}

Instructions:
- Answer using your general knowledge
- Be accurate, helpful, and conversational
- If you don't know something, say so honestly
- Keep responses clear and concise

Answer:"""
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            return {
                'answer': response['response'].strip(),
                'sources': [],
                'type': 'general_knowledge'
            }
    
    def query(self, user_query):
        """
        Point d'entrée principal pour interroger le chatbot
        
        Args:
            user_query: Question de l'utilisateur
            
        Returns:
            Dict avec 'answer', 'sources', 'type'
        """
        result = self.generate_hybrid_response(user_query)
        
        # Sauvegarder dans l'historique
        self.conversation_history.append({
            'user': user_query,
            'bot': result['answer'],
            'type': result['type'],
            'sources': result.get('sources', []),
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def save_conversation(self, filepath=None):
        """
        Sauvegarde la conversation dans un fichier JSON
        
        Args:
            filepath: Chemin du fichier (optionnel, génère automatiquement si None)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversations/chat_{timestamp}.json"
        
        # Créer le dossier conversations s'il n'existe pas
        os.makedirs("conversations", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_conversation(self, filepath):
        """
        Charge une conversation depuis un fichier JSON
        
        Args:
            filepath: Chemin du fichier à charger
            
        Returns:
            Historique de conversation chargé
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)
        
        return self.conversation_history


# Test du chatbot
if __name__ == "__main__":
    print("=== WikiChatbot Test ===\n")
    
    # Créer le chatbot
    bot = WikiChatbot()
    
    # Test 1: Question générale (connaissances du LLM)
    print("Test 1: General Knowledge Question")
    print("Q: What is RAG in AI?")
    result = bot.query("What is RAG in AI?")
    print(f"A: {result['answer'][:200]}...")
    print(f"Type: {result['type']}")
    print(f"Sources: {len(result['sources'])}\n")
    
    # Test 2: Question conversationnelle
    print("Test 2: Casual Question")
    print("Q: Hello! How are you?")
    result = bot.query("Hello! How are you?")
    print(f"A: {result['answer']}")
    print(f"Type: {result['type']}\n")
    
    # Test 3: Question technique générale
    print("Test 3: Technical Question")
    print("Q: How do I read a file in Python?")
    result = bot.query("How do I read a file in Python?")
    print(f"A: {result['answer'][:200]}...")
    print(f"Type: {result['type']}\n")
    
    # Test 4: Question sur la documentation (si disponible)
    print("Test 4: Documentation Question")
    print("Q: What is the system architecture?")
    result = bot.query("What is the system architecture?")
    print(f"A: {result['answer'][:200]}...")
    print(f"Type: {result['type']}")
    print(f"Sources: {len(result['sources'])}\n")
    
    print("=== Test Complete ===")