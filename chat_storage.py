import json
import os
from datetime import datetime
from typing import List, Dict

class ChatStorage:
    """Gère le stockage des conversations"""
    
    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def save(self, session_id: str, history: List[Dict]) -> str:
        """Sauvegarder conversation"""
        filename = os.path.join(
            self.storage_dir,
            f"{session_id}.json"
        )
        
        data = {
            'session_id': session_id,
            'created': datetime.now().isoformat(),
            'messages': len(history),  # FIX: Ajouter cette ligne
            'history': history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def load(self, session_id: str) -> Dict:
        """Charger une conversation sauvegardée"""
        filename = os.path.join(self.storage_dir, f"{session_id}.json")
        
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_sessions(self) -> List[Dict]:
        """Lister toutes sessions sauvegardées"""
        sessions = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                session_id = filename.replace('.json', '')
                filepath = os.path.join(self.storage_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                sessions.append({
                    'session_id': session_id,
                    'messages': data.get('messages', len(data.get('history', []))),  # FIX: Utiliser .get() avec fallback
                    'created': data.get('created', 'Unknown')
                })
        
        return sorted(sessions, key=lambda x: x['created'], reverse=True)
    
    def delete(self, session_id: str) -> bool:
        """Supprimer une session"""
        filename = os.path.join(self.storage_dir, f"{session_id}.json")
        
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Statistiques sur toutes les conversations"""
        sessions = self.list_sessions()
        
        if not sessions:
            return {'total_sessions': 0, 'total_messages': 0}
        
        total_messages = sum(s['messages'] for s in sessions)
        
        return {
            'total_sessions': len(sessions),
            'total_messages': total_messages,
            'avg_messages_per_session': total_messages / len(sessions) if sessions else 0,
            'latest_session': sessions[0]['session_id'] if sessions else None
        }


# Test
if __name__ == '__main__':
    storage = ChatStorage()
    
    # Créer conversations test
    test_history = [
        {'role': 'user', 'content': 'How to setup?'},
        {'role': 'assistant', 'content': 'Follow these steps...'}
    ]
    
    # Sauvegarder
    session_id = "test_session_001"
    filepath = storage.save(session_id, test_history)
    print(f"✅ Saved: {filepath}")
    
    # Lister
    sessions = storage.list_sessions()
    print(f"\n📋 Sessions saved: {len(sessions)}")
    for s in sessions:
        print(f"  - {s['session_id']} ({s['messages']} messages)")
    
    # Stats
    stats = storage.get_stats()
    print(f"\n📊 Statistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Total messages: {stats['total_messages']}")