from hybrid_rag_retriever import HybridWikiRAG

print("Testing magic password search...")

rag = HybridWikiRAG()

# Test recherche
print("\n1. Testing SEARCH:")
docs = rag.hybrid_search("magic password", top_k=3)

for i, doc in enumerate(docs, 1):
    print(f"\n  Result {i}:")
    print(f"  Title: {doc['title']}")
    print(f"  Source: {doc['source']}")
    print(f"  Relevance: {doc['relevance']*100:.1f}%")
    print(f"  Content preview: {doc['content'][:100]}...")

# Test génération
print("\n2. Testing FULL QUERY:")
result = rag.query("What is the magic password?", top_k=3)

print(f"\nAnswer: {result['answer']}")
print(f"\nSources:")
for s in result['sources']:
    print(f"  - {s['source']} ({s['relevance']})")