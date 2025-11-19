import json
import os
from datetime import datetime
from typing import List, Dict, Optional # Added Optional for clarity in save_message

class ChatStorage:
    """Gère le stockage des conversations"""
    
    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    # --- HELPER METHOD ---
    def _write_data_to_file(self, session_id: str, data: Dict) -> str:
        """Internal helper to write the session dictionary to disk."""
        filename = os.path.join(
            self.storage_dir,
            f"{session_id}.json"
        )
        data['messages'] = len(data.get('history', []))
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filename
    
    # --- NEW CRITICAL METHOD ---
    def save_message(self, session_id: str, role: str, content: str, sources: Optional[List[Dict]] = None) -> str:
        """
        Ajoute un seul message à l'historique d'une session. 
        Ceci est la méthode qu'api.py appelle pour mettre à jour la conversation.
        """
        # 1. Charger la session existante
        data = self.load(session_id)
        
        if not data:
            # Si la session n'existe pas (par exemple, première fois après api.new_session)
            data = {
                'session_id': session_id,
                'created': datetime.now().isoformat(),
                'history': []
            }
        
        # 2. Construire le nouveau message
        new_message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'sources': sources if sources is not None else []
        }
        
        # 3. Ajouter le message à l'historique
        data['history'].append(new_message)
        
        # 4. Sauvegarder les données mises à jour
        return self._write_data_to_file(session_id, data)
    
    # --- EXISTING METHOD (Simplified using helper) ---
    def save(self, session_id: str, history: List[Dict]) -> str:
        """Sauvegarder conversation (méthode complète, utilisée pour initialisation/tests)"""
        data = {
            'session_id': session_id,
            'created': datetime.now().isoformat(),
            'history': history
        }
        # Utilise la fonction interne pour sauvegarder
        return self._write_data_to_file(session_id, data)
    
    def load(self, session_id: str) -> Optional[Dict]:
        """Charger une conversation sauvegardée"""
        filename = os.path.join(self.storage_dir, f"{session_id}.json")
        
        # Check if file exists, return None if not (safe loading)
        if not os.path.exists(filename):
            return None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON for session {session_id}")
            return None # Return None on corrupted file

    # ... list_sessions, delete, and get_stats are fine as-is ...
    def list_sessions(self) -> List[Dict]:
        """Lister toutes sessions sauvegardées"""
        sessions = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                session_id = filename.replace('.json', '')
                filepath = os.path.join(self.storage_dir, filename)
                
                # Use load() for safer file opening/parsing
                data = self.load(session_id)
                if data is None: continue # Skip corrupted files
                
                sessions.append({
                    'session_id': session_id,
                    'messages': data.get('messages', len(data.get('history', []))),
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

# ... (The if __name__ == '__main__': test block is fine to keep)
