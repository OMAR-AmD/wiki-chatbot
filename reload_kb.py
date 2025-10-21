from rag_pipeline import RAGPipeline

print("🔄 Reloading knowledge base...")

# Vider l'ancienne collection
rag = RAGPipeline(collection_name="wiki_data")
rag.clear_collection()

# Recharger tous les fichiers
rag.load_from_directory("wiki_data")

print(f"\n✅ Knowledge base reloaded!")
print(f"📊 Total documents in KB: {rag.collection.count()}")

# Tester la recherche
print("\n🔍 Testing search for 'password'...")
results = rag.search("password secret", top_k=1)

if results:
    print(f"✅ Found! Score: {results[0]['score']:.3f}")
    print(f"Content preview: {results[0]['content'][:100]}...")
else:
    print("❌ Nothing found!")