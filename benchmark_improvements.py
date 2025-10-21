import json
import time
from rag_retriever import WikiRAGRetriever
from optimized_rag import OptimizedWikiRAG

test_queries = [
    "How to setup the project locally?",
    "What is the system architecture?",
    "What database do we use?",
    "How do I debug connection errors?",
    "What are the Python naming conventions?"
]

print("="*70)
print("BENCHMARK: BEFORE vs AFTER OPTIMIZATION")
print("="*70)

results = {
    'basic': [],
    'optimized': []
}

# Test version basique (Jour 2)
print("\n1. Testing BASIC RAG (Jour 2)...")
basic_rag = WikiRAGRetriever()

for query in test_queries:
    print(f"  Query: {query[:40]}...", end=" ")
    
    start = time.time()
    result = basic_rag.query(query)
    elapsed = time.time() - start
    
    results['basic'].append({
        'query': query,
        'success': result['success'],
        'time_ms': elapsed * 1000,
        'num_sources': len(result['sources'])
    })
    
    print(f"✓ {elapsed*1000:.0f}ms")

# Test version optimisée
print("\n2. Testing OPTIMIZED RAG (Jour 3)...")
opt_rag = OptimizedWikiRAG()

for query in test_queries:
    print(f"  Query: {query[:40]}...", end=" ")
    
    result = opt_rag.query(query)
    elapsed = result['timing']['total_ms'] / 1000
    
    results['optimized'].append({
        'query': query,
        'success': result['success'],
        'time_ms': result['timing']['total_ms'],
        'num_sources': len(result['sources']),
        'search_ms': result['timing']['search_ms'],
        'generation_ms': result['timing']['generation_ms']
    })
    
    print(f"✓ {result['timing']['total_ms']:.0f}ms")

# Analyse
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

basic_avg = sum(r['time_ms'] for r in results['basic']) / len(results['basic'])
opt_avg = sum(r['time_ms'] for r in results['optimized']) / len(results['optimized'])
improvement = ((basic_avg - opt_avg) / basic_avg) * 100

print(f"\nAverage latency:")
print(f"  Basic RAG: {basic_avg:.0f}ms")
print(f"  Optimized RAG: {opt_avg:.0f}ms")
print(f"  Improvement: {improvement:.1f}% {'faster' if improvement > 0 else 'slower'}")

print(f"\nLatency breakdown (Optimized):")
avg_search = sum(r['search_ms'] for r in results['optimized']) / len(results['optimized'])
avg_gen = sum(r['generation_ms'] for r in results['optimized']) / len(results['optimized'])
print(f"  Search: {avg_search:.0f}ms ({avg_search/opt_avg*100:.0f}%)")
print(f"  Generation: {avg_gen:.0f}ms ({avg_gen/opt_avg*100:.0f}%)")

# Sauvegarder résultats
with open('JOUR3_BENCHMARK.json', 'w') as f:
    json.dump({
        'basic_avg_ms': basic_avg,
        'optimized_avg_ms': opt_avg,
        'improvement_percent': improvement,
        'detailed_results': results
    }, f, indent=2)

print("\n✅ Benchmark results saved to JOUR3_BENCHMARK.json")