import os
import re
from pathlib import Path

def clean_text(text):
    """Nettoie le texte"""
    # Supprimer espaces multiples
    text = re.sub(r'\n\n+', '\n\n', text)
    # Supprimer lignes vides excessives
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def chunk_document(text, chunk_size=500, overlap=50):
    """
    Découpe un document en chunks intelligents
    chunk_size : nombre de mots par chunk
    overlap : nombre de mots qui se répètent entre chunks
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    
    return chunks

def extract_metadata(filename, content):
    """Extrait métadonnées du document"""
    # Trouver le titre (première ligne commençant par #)
    lines = content.split('\n')
    title = None
    for line in lines:
        if line.startswith('#'):
            title = line.replace('#', '').strip()
            break
    
    return {
        'filename': filename,
        'title': title or filename,
        'category': extract_category(filename),
        'length_words': len(content.split())
    }

def extract_category(filename):
    """Extrait catégorie du nom de fichier"""
    categories = {
        'setup': 'Setup & Installation',
        'arch': 'Architecture',
        'api': 'API Documentation',
        'code': 'Coding Standards',
        'troubleshoot': 'Troubleshooting'
    }
    
    for key, category in categories.items():
        if key in filename.lower():
            return category
    return 'General'

# Main
def process_wiki():
    wiki_dir = 'wiki_data'
    output_dir = 'processed_wiki'
    
    # Créer dossier output
    Path(output_dir).mkdir(exist_ok=True)
    
    all_chunks = []
    documents_info = []
    
    # Parcourir tous fichiers
    for filename in os.listdir(wiki_dir):
        if not filename.endswith('.md'):
            continue
        
        filepath = os.path.join(wiki_dir, filename)
        
        print(f"Processing: {filename}")
        
        # Lire fichier
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Nettoyer
        content = clean_text(content)
        
        # Extraire métadonnées
        metadata = extract_metadata(filename, content)
        documents_info.append(metadata)
        
        # Chunker
        chunks = chunk_document(content, chunk_size=500, overlap=50)
        
        # Ajouter métadonnées à chaque chunk
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'content': chunk,
                'source': filename,
                'title': metadata['title'],
                'category': metadata['category'],
                'chunk_id': i
            })
    
    # Sauvegarder chunks
    import json
    with open(f'{output_dir}/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    # Sauvegarder infos documents
    with open(f'{output_dir}/documents_info.json', 'w', encoding='utf-8') as f:
        json.dump(documents_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processing complete!")
    print(f"Total documents: {len(documents_info)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Avg chunk size: {sum(len(c['content'].split()) for c in all_chunks) // len(all_chunks)} words")
    
    return all_chunks, documents_info

if __name__ == '__main__':
    chunks, docs = process_wiki()