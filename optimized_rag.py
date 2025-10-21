from hybrid_rag_retriever import HybridWikiRAG
from config import *
import time
import json
from typing import Dict

class OptimizedWikiRAG(HybridWikiRAG):
    """RAG version optimisée avec config"""
    
    def query(self, query: str, top_k: int = RAG_TOP_K, 
             model: str = LLM_MODEL) -> Dict:
        """Query avec timing et stats"""
        
        start_time = time.time()
        
        # Search
        search_start = time.time()
        docs = self.hybrid_search(query, top_k=top_k, 
                                 use_reranking=USE_RERANKING)
        search_time = time.time() - search_start
        
        if not docs:
            return {
                'answer': "Sorry, I couldn't find relevant information.",
                'sources': [],
                'success': False,
                'timing': {'search_ms': search_time * 1000}
            }
        
        # Generate
        gen_start = time.time()
        answer = self.generate_answer(query, docs, model=model)
        gen_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
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
            'success': True,
            'timing': {
                'search_ms': search_time * 1000,
                'generation_ms': gen_time * 1000,
                'total_ms': total_time * 1000
            }
        }


# Test avec timing
if __name__ == '__main__':
    rag = OptimizedWikiRAG()
    
    print("\n" + "="*70)
    print("OPTIMIZED RAG EVALUATION")
    print("="*70)
    
    test_queries = [
        "How to setup locally?",
        "What is our database?",
        "How do I debug errors?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.query(query)
        
        if result['success']:
            print(f"Answer: {result['answer'][:150]}...")
            print(f"\nTiming:")
            print(f"  Search: {result['timing']['search_ms']:.0f}ms")
            print(f"  Generation: {result['timing']['generation_ms']:.0f}ms")
            print(f"  Total: {result['timing']['total_ms']:.0f}ms")
            print(f"\nSources:")
            for s in result['sources']:
                print(f"  - {s['title']} ({s['relevance']})")