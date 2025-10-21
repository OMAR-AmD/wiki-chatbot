import json

# Charger chunks
with open('processed_wiki/chunks.json', 'r') as f:
    chunks = json.load(f)

print("="*60)
print("CHUNK QUALITY ANALYSIS")
print("="*60)

# Stats générales
total_chunks = len(chunks)
total_words = sum(len(c['content'].split()) for c in chunks)
avg_words = total_words // total_chunks

print(f"\nTotal chunks: {total_chunks}")
print(f"Total words: {total_words}")
print(f"Average chunk size: {avg_words} words")

# Par catégorie
from collections import defaultdict
by_category = defaultdict(int)
for chunk in chunks:
    by_category[chunk['category']] += 1

print(f"\nChunks par catégorie:")
for cat, count in sorted(by_category.items()):
    print(f"  {cat}: {count}")

# Exemple de chunks
print(f"\n{'='*60}")
print("EXEMPLE DE 3 CHUNKS")
print(f"{'='*60}")

for i in range(min(3, len(chunks))):
    chunk = chunks[i]
    print(f"\nChunk {i+1}:")
    print(f"Source: {chunk['source']}")
    print(f"Title: {chunk['title']}")
    print(f"Category: {chunk['category']}")
    print(f"Content preview: {chunk['content'][:150]}...")
    print("-"*60)