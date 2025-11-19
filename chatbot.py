# In chatbot.py

import ollama
from rag_pipeline import RAGPipeline
from chat_storage import ChatStorage
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class WikiChatbot:
    def __init__(self, model_name="llama2"):
        """
        Initialise le chatbot Wiki
        """
        self.model_name = model_name
        # NOTE: If self.rag initialization fails (e.g., ChromaDB error), the server will crash here.
        self.rag = RAGPipeline()
        self.storage = ChatStorage()
        self.system_prompt = (
            "You are a helpful Wiki Chatbot. "
            "Use the provided context to answer the user's questions accurately. "
            "If the answer is not in the context, state that clearly."
        )

    def format_history(self, history: List[Dict]) -> List[Dict]:
        """Formats history from ChatStorage format to Ollama messages list."""
        return [{'role': msg['role'], 'content': msg['content']} for msg in history]


    def query(self, user_query: str, session_id: Optional[str] = None) -> Dict:
        
        # --- RAG RETRIEVAL ---
        # 1. Perform RAG search on the NEW user query
        retrieval_result = self.rag.search(user_query)
        
        # 🎯 FIX FOR ATTRIBUTE ERROR (list.get):
        # Assuming RAGPipeline.search() has been fixed to return a dictionary, 
        # but adding safety check just in case it returns None or an empty list.
        if not isinstance(retrieval_result, dict):
            retrieval_result = {"context": "No RAG data found (Retrieval failed).", "sources": [], "type": "no_rag"}
        
        sources = retrieval_result.get('sources', [])
        rag_context = retrieval_result.get('context', 'No context available.')
        
        # --- HISTORY LOADING ---
        messages = []
        
        # 2. Load History only if session_id is provided
        if session_id:
            try:
                session_data = self.storage.load(session_id)
                
                # Use .get() for safety in case 'history' key is missing, defaults to empty list
                full_history = session_data.get('history', []) if session_data else []
                
                # history_for_prompt contains all messages *prior* to the current turn.
                history_for_prompt = full_history[:-1] 
                
                # Convert stored history into the Ollama messages list format
                if history_for_prompt:
                    messages.extend(self.format_history(history_for_prompt))
            except Exception as e:
                logger.error(f"Error loading chat history for session {session_id}: {e}")
                # Continue without history if loading fails
        
        # --- PROMPT CONSTRUCTION ---
        
        # 3. Add the SYSTEM prompt with the RAG Context (prepended as the first element)
        contextualized_system_prompt = (
            f"{self.system_prompt}\n\n"
            f"--- RAG CONTEXT ---\n{rag_context}\n-----------------"
        )
        messages.insert(0, {'role': 'system', 'content': contextualized_system_prompt})

        # 4. Add the FINAL user query (appended as the last element)
        messages.append({'role': 'user', 'content': user_query})

        # --- OLLAMA CALL ---
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages # Pass the full list: [System, History..., Final User Query]
            )
            
            return {
                'answer': response['message']['content'],
                'sources': sources,
                'type': retrieval_result.get('type', 'hybrid_rag'),
            }
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            return {
                'answer': "Sorry, I ran into a problem communicating with the language model. Please check the Ollama server.",
                'sources': [],
                'type': 'error',
            }
