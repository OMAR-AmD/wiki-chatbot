import json
from sentence_transformers import SentenceTransformer
import chromadb

print("Testing Search Functionality\n")

# Load vector store
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection("wiki_documents")

# Load embeddings model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Test queries
test_queries = [
    "How do I setup the project locally?",
    "What is the system architecture?",
    "What database do we use?",
    "How do I debug connection errors?",
    "What are the Python coding standards?"
]

print("="*70)
print("SEARCH TEST RESULTS")
print("="*70)

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-"*70)
    
    # Generate query embedding
    query_embedding = model.encode(query)
    
    # Search in Chroma
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,  # Top 3 results
        include=["documents", "metadatas", "distances"]
    )
    
    # Display results
    if results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (0-1)
            similarity = 1 - (distance / 2)  # Cosine distance -> similarity
            
            print(f"\n  Result {i+1}:")
            print(f"    Source: {metadata.get('source', 'N/A')}")
            print(f"    Title: {metadata.get('title', 'N/A')}")
            print(f"    Category: {metadata.get('category', 'N/A')}")
            print(f"    Relevance: {similarity:.2%}")
            print(f"    Preview: {doc[:100]}...")
    else:
        print("  No results found")

print("\n" + "="*70)
print("✅ Search test complete!")