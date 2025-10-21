import os
import json
from pathlib import Path

def chunk_text(text, chunk_size=500, overlap=50):
    """Découpe le texte en chunks avec overlap"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def detect_category(filename, text):
    """Détecte la catégorie basée sur le nom de fichier et contenu"""
    filename_lower = filename.lower()
    text_lower = text.lower()
    
    # Par nom de fichier
    if any(word in filename_lower for word in ['setup', 'install', 'config']):
        return 'setup'
    elif any(word in filename_lower for word in ['api', 'endpoint']):
        return 'api'
    elif any(word in filename_lower for word in ['database', 'db']):
        return 'database'
    elif any(word in filename_lower for word in ['troubleshoot', 'debug', 'error']):
        return 'troubleshooting'
    
    # Par contenu
    if any(word in text_lower for word in ['setup', 'install', 'configuration']):
        return 'setup'
    elif any(word in text_lower for word in ['api', 'endpoint', 'request']):
        return 'api'
    elif any(word in text_lower for word in ['database', 'sql', 'query']):
        return 'database'
    elif any(word in text_lower for word in ['error', 'debug', 'troubleshoot']):
        return 'troubleshooting'
    
    return 'general'

def extract_title(chunk, filename):
    """Extrait un titre du chunk ou utilise le nom de fichier"""
    # Chercher un titre markdown (# Title)
    lines = chunk.split('\n')
    for line in lines:
        if line.startswith('#'):
            title = line.replace('#', '').strip()
            if title:
                return title[:80]
    
    # Sinon, utiliser le début du texte
    words = chunk.split()[:10]
    title = ' '.join(words)
    
    if len(title) > 80:
        title = title[:77] + "..."
    
    return title if title else filename

def process_wiki_documents():
    """Process tous les documents markdown du wiki"""
    
    print("=" * 70)
    print("WIKI DOCUMENT PROCESSOR")
    print("=" * 70)
    
    wiki_dir = 'wiki_data'
    output_dir = 'processed_wiki'
    
    # 1. Vérifier que le dossier existe
    if not os.path.exists(wiki_dir):
        print(f"❌ Error: Directory '{wiki_dir}' not found!")
        print("\nPlease create the 'wiki_data' directory and add your .md files")
        return
    
    # 2. Lister les fichiers markdown
    print(f"\n1. Scanning directory: {wiki_dir}")
    md_files = [f for f in os.listdir(wiki_dir) if f.endswith('.md')]
    
    if not md_files:
        print(f"❌ No .md files found in {wiki_dir}")
        print("\nPlease add markdown files to the wiki_data directory")
        return
    
    print(f"   ✅ Found {len(md_files)} markdown files")
    for f in md_files:
        print(f"      - {f}")
    
    # 3. Créer dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Traiter chaque fichier
    print("\n2. Processing documents...")
    
    all_chunks = []
    documents_info = []
    
    for filename in sorted(md_files):
        filepath = os.path.join(wiki_dir, filename)
        
        print(f"\n   Processing: {filename}")
        
        # Lire le fichier
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        
        word_count = len(content.split())
        print(f"      Words: {word_count}")
        
        # Chunker le contenu
        doc_chunks = chunk_text(content, chunk_size=500, overlap=50)
        print(f"      Chunks: {len(doc_chunks)}")
        
        # Détecter catégorie
        category = detect_category(filename, content)
        
        # Traiter chaque chunk
        for i, chunk_content in enumerate(doc_chunks):
            title = extract_title(chunk_content, filename)
            
            all_chunks.append({
                'content': chunk_content,
                'source': filename,
                'title': title,
                'category': category,
                'chunk_id': i
            })
        
        documents_info.append({
            'filename': filename,
            'category': category,
            'chunks': len(doc_chunks),
            'words': word_count
        })
    
    # 5. Sauvegarder les chunks
    print("\n3. Saving processed data...")
    
    chunks_file = f'{output_dir}/chunks.json'
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved {len(all_chunks)} chunks to {chunks_file}")
    
    # 6. Sauvegarder les métadonnées
    metadata_file = f'{output_dir}/metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(md_files),
            'total_chunks': len(all_chunks),
            'documents': documents_info
        }, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved metadata to {metadata_file}")
    
    # 7. Statistiques
    print("\n4. Statistics:")
    
    categories = {}
    for chunk in all_chunks:
        cat = chunk['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n   Total documents: {len(md_files)}")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Average chunks per document: {len(all_chunks) / len(md_files):.1f}")
    
    print("\n   Categories distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_chunks)) * 100
        print(f"      - {cat}: {count} chunks ({percentage:.1f}%)")
    
    # 8. Résumé
    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nProcessed files: {len(md_files)}")
    print(f"Output directory: {output_dir}/")
    print(f"Total chunks: {len(all_chunks)}")
    
    print("\nNext step: Run 'python wiki_embedder.py' to create embeddings")
    
    return all_chunks


if __name__ == '__main__':
    process_wiki_documents()