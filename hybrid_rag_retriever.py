import os
import json
import math
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama

# ==============================================================================
# CHEMINS LOCAUX (modifie si ton username n'est pas 'omara')
# ==============================================================================
LOCAL_EMBEDDING_PATH = r"C:\Users\omara\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
LOCAL_RERANKER_PATH = r"C:\Users\omara\.cache\huggingface\hub\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8"

# Force mode HORS LIGNE
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'

# ==============================================================================
# CLASSE HYBRIDWIKIRAG
# ==============================================================================

class HybridWikiRAG:
    """RAG avec recherche hybride (dense + sparse) + reranking"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print("Initializing Hybrid WikiRAG with Reranking...")
        
        # Vector store (dense search)
        self.client = chromadb.PersistentClient(path="./chroma_data")
        
        try:
            self.collection = self.client.get_collection("wiki")
            print(f"✅ Loaded existing collection with {self.collection.count()} documents")
        except Exception as e:
            print(f"❌ Error loading collection: {e}")
            print("Run 'python create_vector_store.py' first!")
            raise
        
        # Charger embedding model
        print("Loading embedding model from local path...")
        self.embedding_model = SentenceTransformer(LOCAL_EMBEDDING_PATH)
        
        # Charger documents pour sparse search
        print("Loading documents for sparse search...")
        with open('processed_wiki/chunks.json', 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        self.corpus = [c['content'] for c in self.chunks]
        
        # TF-IDF vectorizer
        print("Building TF-IDF index...")
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus)
        
        # Reranker model
        print("Loading reranker model from local path...")
        self.reranker = CrossEncoder(LOCAL_RERANKER_PATH)
        
        print("✅ Hybrid WikiRAG with Reranking initialized\n")
    
    def dense_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche dense (embeddings)"""
        query_embedding = self.embedding_model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        dense_results = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            similarity = 1 - (distance / 2)
            
            dense_results.append({
                'content': doc,
                'source': metadata['source'],
                'title': metadata['title'],
                'category': metadata.get('category', 'General'),
                'relevance': similarity,
                'method': 'dense'
            })
        
        return dense_results
    
    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche sparse (TF-IDF)"""
        query_vec = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        sparse_results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                chunk = self.chunks[idx]
                sparse_results.append({
                    'content': chunk['content'],
                    'source': chunk['source'],
                    'title': chunk['title'],
                    'category': chunk.get('category', 'General'),
                    'relevance': float(similarities[idx]),
                    'method': 'sparse'
                })
        
        return sparse_results
    
    def sigmoid(self, x: float) -> float:
        """Fonction sigmoid pour normaliser"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def rerank_results(self, query: str, documents: List[Dict], 
                       top_k: int = 3) -> List[Dict]:
        """Rerank avec cross-encoder"""
        if not documents:
            return []
        
        print(f"  🔄 Reranking {len(documents)} documents...")
        pairs = [(query, doc['content'][:512]) for doc in documents]
        raw_scores = self.reranker.predict(pairs)
        normalized_scores = [self.sigmoid(float(score)) for score in raw_scores]
        
        for i, doc in enumerate(documents):
            doc['rerank_score'] = normalized_scores[i]
        
        reranked = sorted(documents, 
                         key=lambda x: x['rerank_score'], 
                         reverse=True)[:top_k]
        
        print(f"  ✅ Reranked to top {len(reranked)} documents")
        
        return reranked
    
    def hybrid_search(self, query: str, top_k: int = 3, 
                      dense_weight: float = 0.7, 
                      sparse_weight: float = 0.3,
                      use_reranking: bool = True) -> List[Dict]:
        """Recherche hybride (dense + sparse + reranking)"""
        search_k = top_k * 3 if use_reranking else top_k * 2
        
        dense_results = self.dense_search(query, top_k=search_k)
        sparse_results = self.sparse_search(query, top_k=search_k)
        
        combined = {}
        
        # Combine dense results
        for result in dense_results:
            key = f"{result['source']}_{result['title']}"
            combined[key] = {
                **result,
                'hybrid_score': result['relevance'] * dense_weight
            }
        
        # Combine sparse results
        for result in sparse_results:
            key = f"{result['source']}_{result['title']}"
            if key in combined:
                combined[key]['hybrid_score'] += result['relevance'] * sparse_weight
                combined[key]['method'] = 'hybrid'
            else:
                combined[key] = {
                    **result,
                    'hybrid_score': result['relevance'] * sparse_weight,
                    'method': 'sparse_only'
                }
        
        # Sort by hybrid score
        sorted_results = sorted(combined.values(), 
                               key=lambda x: x['hybrid_score'], 
                               reverse=True)[:search_k]
        
        # Apply reranking
        if use_reranking and sorted_results:
            sorted_results = self.rerank_results(query, sorted_results, top_k)
            for r in sorted_results:
                r['relevance'] = r['rerank_score']
                r['method'] = r['method'] + '+rerank'
        else:
            sorted_results = sorted_results[:top_k]
            for r in sorted_results:
                r['relevance'] = r['hybrid_score']
        
        return sorted_results
    
    def generate_answer(self, query: str, context: List[Dict], 
                        model: str = 'llama2') -> str:
        """Génère réponse avec Ollama"""
        context_text = "\n\n".join([
            f"[{doc['title']}]\n{doc['content']}"
            for doc in context
        ])
        
        prompt = f"""You are a helpful wiki assistant. Answer based on the provided documentation.

WIKI DOCUMENTATION:
{context_text}

USER QUESTION: {query}

Instructions:
- Answer concisely and accurately based ONLY on the documentation above
- If the answer is not in the documentation, say "This information is not available in the wiki"
- Provide specific details from the documentation
- Keep your answer clear and to the point"""
        
        print(f"  🤖 Generating answer with {model}...")
        
        try:
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def query(self, query: str, top_k: int = 3, 
             model: str = 'llama2',
             use_reranking: bool = True) -> Dict:
        """Pipeline complet RAG"""
        
        print(f"\n{'='*70}")
        print(f"🔍 Processing query: {query}")
        print('='*70)
        
        start_time = time.time()
        
        # Search
        docs = self.hybrid_search(query, top_k=top_k, use_reranking=use_reranking)
        
        if not docs:
            return {
                'answer': "Sorry, I couldn't find relevant information in the wiki.",
                'sources': [],
                'success': False
            }
        
        # Generate answer
        answer = self.generate_answer(query, docs, model=model)
        
        elapsed = time.time() - start_time
        
        return {
            'answer': answer,
            'sources': [
                {
                    'title': doc['title'],
                    'source': doc['source'],
                    'category': doc['category'],
                    'relevance': f"{doc['relevance']*100:.1f}%",
                    'method': doc['method']
                }
                for doc in docs
            ],
            'success': True,
            'time_seconds': round(elapsed, 2)
        }


# ==============================================================================
# TESTS
# ==============================================================================

def test_search_only():
    """Test recherche uniquement (sans génération)"""
    print("\n" + "="*70)
    print("TEST: SEARCH ONLY")
    print("="*70)
    
    rag = HybridWikiRAG()
    
    test_queries = [
        "How to setup locally?",
        "What database do we use?",
        "Architecture overview"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        docs = rag.hybrid_search(query, top_k=3)
        
        print(f"\n📚 Top 3 Results:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. {doc['title']}")
            print(f"     Source: {doc['source']}")
            print(f"     Relevance: {doc['relevance']*100:.1f}%")
            print(f"     Method: {doc['method']}")


def test_full_pipeline():
    """Test pipeline complet avec génération"""
    print("\n" + "="*70)
    print("TEST: FULL PIPELINE")
    print("="*70)
    
    rag = HybridWikiRAG()
    
    test_queries = [
        "How do I setup the project locally?",
        "What database do we use?"
    ]
    
    for query in test_queries:
        result = rag.query(query, top_k=2, use_reranking=True)
        
        if result['success']:
            print(f"\n📝 ANSWER:")
            print(f"{result['answer']}\n")
            
            print(f"📚 SOURCES:")
            for src in result['sources']:
                print(f"  • {src['title']}")
                print(f"    Relevance: {src['relevance']} | Method: {src['method']}")
            
            print(f"\n⏱️  Time: {result['time_seconds']}s")
        else:
            print(f"\n❌ No answer found")


if __name__ == '__main__':
    import sys
    
    print("--- HYBRID RAG RETRIEVER ---\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--full':
            test_full_pipeline()
        elif sys.argv[1] == '--search':
            test_search_only()
        else:
            print("Usage:")
            print("  python hybrid_rag_retriever.py --search  (test search only)")
            print("  python hybrid_rag_retriever.py --full    (test full pipeline)")
    else:
        print("ℹ️  Run with:")
        print("  python hybrid_rag_retriever.py --search  (test search only)")
        print("  python hybrid_rag_retriever.py --full    (test full pipeline)")