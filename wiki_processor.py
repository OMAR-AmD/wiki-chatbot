import os
import json
from pathlib import Path

def detect_category(filename: str, content: str) -> str:
    """Détecte la catégorie du document"""
    filename_lower = filename.lower()
    content_lower = content.lower()
    
    if 'api' in filename_lower or 'endpoint' in content_lower:
        return 'api'
    elif 'database' in filename_lower or 'schema' in filename_lower:
        return 'database'
    elif 'setup' in filename_lower or 'installation' in content_lower:
        return 'setup'
    elif 'deploy' in filename_lower:
        return 'deployment'
    elif 'security' in filename_lower or 'securite' in filename_lower:
        return 'security'
    elif 'test' in filename_lower:
        return 'testing'
    elif 'troubleshoot' in filename_lower or 'debug' in content_lower:
        return 'troubleshooting'
    elif 'monitor' in filename_lower:
        return 'monitoring'
    else:
        return 'general'

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Découpe le texte en chunks avec overlap
    
    Args:
        text: Texte à découper
        chunk_size: Taille max d'un chunk (en caractères)
        overlap: Nombre de caractères à chevaucher
        
    Returns:
        Liste de chunks
    """
    # Split par mots
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Prendre chunk_size mots
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Avancer avec overlap
        start = end - overlap
        
        if start >= len(words):
            break
    
    return chunks

def extract_title(content: str, filename: str) -> str:
    """Extrait le titre du document"""
    lines = content.split('\n')
    
    # Chercher première ligne avec #
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            return line.replace('#', '').strip()
    
    # Sinon utiliser le nom du fichier
    return filename.replace('.md', '').replace('_', ' ').title()

def process_wiki_documents():
    """Traiter tous les documents wiki"""
    wiki_dir = 'wiki_data'
    output_dir = 'processed_wiki'
    
    # Créer répertoire de sortie
    Path(output_dir).mkdir(exist_ok=True)
    
    print("="*70)
    print("WIKI DOCUMENT PROCESSOR")
    print("="*70)
    
    # 1. Scanner le répertoire
    print(f"\n1. Scanning directory: {wiki_dir}")
    
    if not os.path.exists(wiki_dir):
        print(f"   ❌ Error: Directory '{wiki_dir}' does not exist!")
        return
    
    md_files = [f for f in os.listdir(wiki_dir) if f.endswith('.md')]
    
    if not md_files:
        print(f"   ❌ Error: No .md files found in '{wiki_dir}'!")
        return
    
    print(f"   ✅ Found {len(md_files)} markdown files")
    for f in md_files:
        print(f"      - {f}")
    
    # 2. Traiter chaque document
    print(f"\n2. Processing documents...")
    
    all_chunks = []
    documents_info = []
    
    for filename in sorted(md_files):
        filepath = os.path.join(wiki_dir, filename)
        
        print(f"\n   Processing: {filename}")
        
        # Lire le fichier
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier que le contenu n'est pas vide
        if not content.strip():
            print(f"      ⚠️  Warning: Empty file, skipping...")
            continue
        
        # Compter mots
        word_count = len(content.split())
        print(f"      Words: {word_count}")
        
        # Détecter catégorie
        category = detect_category(filename, content)
        
        # Extraire titre
        title = extract_title(content, filename)
        
        # Découper en chunks
        chunks = chunk_text(content, chunk_size=500, overlap=50)
        print(f"      Chunks: {len(chunks)}")
        
        # Ajouter chunks à la liste
        for i, chunk_content in enumerate(chunks):
            all_chunks.append({
                'content': chunk_content,
                'source': filename,
                'title': title,
                'category': category,
                'chunk_id': i
            })
        
        # Info du document
        documents_info.append({
            'filename': filename,
            'title': title,
            'category': category,
            'word_count': word_count,
            'chunk_count': len(chunks)
        })
    
    # 3. Sauvegarder les résultats
    print(f"\n3. Saving processed data...")
    
    # Sauvegarder chunks
    chunks_file = os.path.join(output_dir, 'chunks.json')
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved {len(all_chunks)} chunks to {chunks_file}")
    
    # Sauvegarder metadata
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(documents_info),
            'total_chunks': len(all_chunks),
            'documents': documents_info
        }, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved metadata to {metadata_file}")
    
    # 4. Statistiques
    print(f"\n4. Statistics:")
    print(f"\n   Total documents: {len(documents_info)}")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Average chunks per document: {len(all_chunks)/len(documents_info):.1f}")
    
    # Compter par catégorie
    category_counts = {}
    for chunk in all_chunks:
        cat = chunk['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\n   Categories distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      - {cat}: {count} chunks ({count/len(all_chunks)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nProcessed files: {len(documents_info)}")
    print(f"Output directory: {output_dir}/")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"\nNext step: Run 'python wiki_embedder.py' to create embeddings")

if __name__ == '__main__':
    process_wiki_documents()