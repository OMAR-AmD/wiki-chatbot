import os
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'
import streamlit as st
import time
from chatbot import WikiChatbot
from chat_storage import ChatStorage
import json
from datetime import datetime

# Configuration Streamlit
st.set_page_config(
    page_title="Wiki Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - CORRIGÉ
st.markdown("""
    <style>
    .chatbot-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1a1a1a;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1a1a1a;
        border-left: 4px solid #667eea;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: #1a1a1a;
        border-left: 3px solid #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = WikiChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'storage' not in st.session_state:
    st.session_state.storage = ChatStorage()

storage = st.session_state.storage

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Section: Session Management
    st.subheader("Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 New Chat"):
            st.session_state.chatbot = WikiChatbot()
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat"):
            filepath = st.session_state.chatbot.save_conversation()
            st.success(f"Saved to {filepath}")
    
    # Section: Conversation History
    st.subheader("Conversation History")
    
    sessions = storage.list_sessions()
    if sessions:
        st.write(f"📊 {len(sessions)} saved conversations")
        
        selected_session = st.selectbox(
            "Load conversation:",
            options=[s['session_id'] for s in sessions],
            format_func=lambda x: f"{x} ({next((s['messages'] for s in sessions if s['session_id'] == x), 0)} messages)"
        )
        
        if st.button("Load"):
            data = storage.load(selected_session)
            if data:
                st.session_state.messages = data['history']
                st.success("Conversation loaded!")
                st.rerun()
        
        if st.button("Delete"):
            storage.delete(selected_session)
            st.success("Conversation deleted!")
            st.rerun()
    
    # Section: Statistics
    st.subheader("Statistics")
    stats = storage.get_stats()
    st.metric("Total Conversations", stats['total_sessions'])
    st.metric("Total Messages", stats['total_messages'])
    
    # Section: About
    st.subheader("About")
    st.info("""
    **Wiki Chatbot** 🤖
    
    A local AI chatbot powered by Llama 2
    using Retrieval-Augmented Generation (RAG).
    
    All data stays on your machine.
    """)

# Main Chat Interface
st.title("🤖 Wiki Chatbot")
st.markdown("*Ask anything about the documentation*")

# Display conversation
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        if role == 'user':
            st.markdown(f"""
            <div class="user-message">
            <b>You:</b> {content}
            </div>
            """, unsafe_allow_html=True)
        elif role == 'assistant':
            st.markdown(f"""
            <div class="bot-message">
            <b>Bot:</b> {content}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if available
            if 'sources' in message and message['sources']:
                with st.expander("📚 Sources"):
                    for source in message['sources']:
                        st.markdown(f"""
                        <div class="source-box">
                        <b>{source.get('title', 'Unknown')}</b> ({source.get('relevance', 'N/A')}) 
                        <br>
                        <small>From: {source.get('source', 'Unknown')}</small>
                        </div>
                        """, unsafe_allow_html=True)

# Input area
st.markdown("---")

col1, col2 = st.columns([0.9, 0.1])

with col1:
    user_input = st.text_input(
        "You:",
        placeholder="Ask me anything about the wiki...",
        key="user_input"
    )

with col2:
    send_button = st.button("Send", key="send_button")

# Process input
if user_input and send_button:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    # Show thinking indicator
    with st.spinner("🤔 Thinking..."):
        start_time = time.time()
        result = st.session_state.chatbot.query(user_input)
        elapsed = time.time() - start_time
    
    # Add bot message
    st.session_state.messages.append({
        'role': 'assistant',
        'content': result['answer'],
        'timestamp': datetime.now().isoformat(),
        'sources': result.get('sources', []),
        'latency_ms': elapsed * 1000
    })
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Response Time", f"{elapsed*1000:.0f}ms")
    with col2:
        st.metric("Sources Found", len(result.get('sources', [])))
    with col3:
        # Compte le nombre de messages utilisateur
        user_count = sum(1 for m in st.session_state.messages if m.get('role') == 'user')
        st.metric("Turn", user_count)
    
    # Auto-save
    st.session_state.chatbot.conversation_history = st.session_state.messages
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
<p>Wiki Chatbot v1.0 | Powered by Llama 2 + Local RAG</p>
<p>All conversations are saved locally on your machine</p>
</div>
""", unsafe_allow_html=True)