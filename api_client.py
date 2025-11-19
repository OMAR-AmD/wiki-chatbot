import requests
from typing import Dict, List, Optional # Corrected import

class WikiChatbotAPIClient:
    """Client pour API Wiki Chatbot"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
    
    # Corrected: Added session_id argument and included it in the payload
    def query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Send query to API, optionally including session_id for context"""
        
        payload = {"query": query}
        if session_id:
            payload["session_id"] = session_id
            
        response = requests.post(
            f"{self.base_url}/query",
            json=payload # Send the payload dictionary
        )
        response.raise_for_status()
        return response.json()
    
    def new_session(self) -> Dict:
        """Start new session"""
        response = requests.post(f"{self.base_url}/session/new")
        response.raise_for_status()
        return response.json()
    
    def list_sessions(self) -> List[Dict]:
        """List all sessions"""
        response = requests.get(f"{self.base_url}/session/list")
        response.raise_for_status()
        return response.json()
    
    def get_session(self, session_id: str) -> Dict:
        """Get specific session"""
        response = requests.get(f"{self.base_url}/session/{session_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, session_id: str) -> Dict:
        """Delete session"""
        response = requests.delete(f"{self.base_url}/session/{session_id}")
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        response = requests.get(f"{self.base_url}/statistics")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
