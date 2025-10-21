import requests
from typing import Dict, List

class WikiChatbotAPIClient:
    """Client pour API Wiki Chatbot"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
    
    def query(self, query: str) -> Dict:
        """Send query to API"""
        response = requests.post(
            f"{self.base_url}/query",
            json={"query": query}
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