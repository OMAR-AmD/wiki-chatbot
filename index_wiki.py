import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

print("Starting Wiki Indexing...\n")

# Configuration
WIKI_DATA_DIR = "./wiki_data"
CHROMA_DIR = "./chroma_data"
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete existing collection if it exists
try:
    client.delete_collection("wiki_documents")
    print("Deleted existing collection")
except:
    pass

# Create new collection
collection = client.get_or_create_collection("wiki_documents")

# Load embedding model
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Track indexing stats
doc_id = 0
indexed_files = 0

# Process markdown files
print(f"\nScanning {WIKI_DATA_DIR} for markdown files...\n")

if not os.path.exists(WIKI_DATA_DIR):
    print(f"❌ Error: {WIKI_DATA_DIR} directory not found!")
    exit()

for root, dirs, files in os.walk(WIKI_DATA_DIR):
    for file in files:
        if file.endswith('.md'):
            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, WIKI_DATA_DIR)
            
            print(f"Processing: {relative_path}")
            
            try:
                # Read markdown file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title from filename or first heading
                title = file.replace('.md', '').replace('_', ' ').title()
                
                # Try to extract title from first heading
                heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if heading_match:
                    title = heading_match.group(1)
                
                # Extract category from folder structure
                category = os.path.basename(os.path.dirname(filepath))
                if category == 'wiki_data':
                    category = 'General'
                
                # Split content into chunks (by paragraphs/sections)
                # This prevents huge documents and creates multiple searchable chunks
                chunks = content.split('\n\n')
                chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                
                # Index each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk) > 20:  # Skip very small chunks
                        doc_id += 1
                        
                        # Generate embedding
                        embedding = model.encode(chunk)
                        
                        # Add to collection
                        collection.add(
                            ids=[str(doc_id)],
                            embeddings=[embedding.tolist()],
                            documents=[chunk],
                            metadatas=[{
                                'source': relative_path,
                                'title': title,
                                'category': category,
                                'chunk': chunk_idx
                            }]
                        )
                
                indexed_files += 1
                print(f"  ✓ Added {len(chunks)} chunks\n")
                
            except Exception as e:
                print(f"  ❌ Error processing {filepath}: {e}\n")

# Print summary
print("="*70)
print("INDEXING COMPLETE")
print("="*70)
print(f"Files indexed: {indexed_files}")
print(f"Total chunks: {doc_id}")
print(f"Collection: wiki_documents")
print(f"Database: {CHROMA_DIR}")
print("="*70)

if doc_id > 0:
    print("✅ Wiki documents successfully indexed!")
else:
    print("❌ No documents were indexed. Check your wiki_data folder.")