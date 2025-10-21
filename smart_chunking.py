import re
import json
from pathlib import Path

def smart_chunk_document(content: str, source: str) -> list:
    """
    Chunk document par sections intelligemment
    """
    chunks = []
    
    # Split par sections (# ou ##)
    sections = re.split(r'(^#{1,2}\s+.+$)', content, flags=re.MULTILINE)
    
    current_chunk = ""
    current_title = source
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        
        # Détecte titre
        if section.startswith('#'):
            if current_chunk.strip():  # Sauvegarder chunk précédent
                chunks.append({
                    'content': current_chunk.strip(),
                    'title': current_title
                })
            current_title = section.replace('#', '').strip()
            current_chunk = ""
        else:
            current_chunk += section
    
    # Last chunk
    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip(),
            'title': current_title
        })
    
    return chunks


# Reprocess data with smart chunking
def reprocess_wiki_smart():
    wiki_dir = 'wiki_data'
    output_dir = 'processed_wiki'
    
    all_chunks = []
    documents_info = []
    
    for filename in sorted(os.listdir(wiki_dir)):
        if not filename.endswith('.md'):
            continue
        
        filepath = os.path.join(wiki_dir, filename)
        
        print(f"Processing with smart chunking: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Smart chunk
        chunks = smart_chunk_document(content, filename)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'content': chunk['content'],
                'source': filename,
                'title': chunk['title'],
                'chunk_id': i
            })
        
        documents_info.append({
            'filename': filename,
            'chunks': len(chunks),
            'total_words': len(content.split())
        })
    
    # Sauvegarder
    with open(f'{output_dir}/chunks_smart.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Smart chunking complete!")
    print(f"Total chunks: {len(all_chunks)} (vs {len([c for c in json.load(open(f'{output_dir}/chunks.json', encoding='utf-8'))])} before)")
    
    return all_chunks

if __name__ == '__main__':
    import os
    chunks = reprocess_wiki_smart()