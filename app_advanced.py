import os
import streamlit as st
from chatbot import WikiChatbot
from chat_storage import ChatStorage
import json
from datetime import datetime
import time

# ==============================================================================
# ⚠️ CORRECTION CRITIQUE HORS LIGNE (À placer en premier)
# ==============================================================================
# Force les librairies Hugging Face (Sentence-Transformers/CrossEncoder) à
# n'utiliser que le cache local, ignorant toute tentative de connexion réseau.
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'

st.set_page_config(
    page_title="Wiki Chatbot Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if 'chatbot' not in st.session_state:
    # L'initialisation de WikiChatbot se fait ici, forçant le chargement des modèles
    # avec les chemins locaux définis dans hybrid_rag_retriever.py
    st.session_state.chatbot = WikiChatbot()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'storage' not in st.session_state:
    st.session_state.storage = ChatStorage()

storage = st.session_state.storage

# Sidebar with tabs
with st.sidebar:
    st.title("🤖 Wiki Chatbot Pro")
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Chat Options")
        if st.button("🔄 New Conversation"):
            st.session_state.chatbot = WikiChatbot()
            st.session_state.messages = []
            st.rerun()
        
        if st.button("💾 Save Conversation"):
            filepath = st.session_state.chatbot.save_conversation()
            st.success(f"✅ Saved!")
    
    with tab2:
        st.subheader("Statistics")
        stats = storage.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📁 Total Conversations", stats['total_sessions'])
        with col2:
            st.metric("💬 Total Messages", stats['total_messages'])
        
        # Liste sessions
        if stats['total_sessions'] > 0:
            st.write("**Recent Conversations:**")
            sessions = storage.list_sessions()[:5]
            for s in sessions:
                st.write(f"• {s['session_id']}: {s['messages']} messages")
    
    with tab3:
        st.subheader("Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Theme", ["Light", "Dark"])
        with col2:
            model = st.selectbox("LLM Model", ["llama2", "mistral"])
        
        st.write("**Advanced Options:**")
        debug_mode = st.checkbox("Debug Mode")

# Main content
col1, col2 = st.columns([0.7, 0.3])

with col1:
    st.title("💬 Chat Interface")
    
    # Chat display
    chat_container = st.container(height=400)
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.write(f"**You:** {msg['content']}")
            else:
                st.write(f"**Bot:** {msg['content']}")
                if 'latency_ms' in msg:
                    st.caption(f"⏱️ {msg['latency_ms']:.0f}ms")
    
    # Input
    user_input = st.text_input("Message:", key="msg_input")
    
    if user_input:
        # Add to history
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get response
        with st.spinner("Processing..."):
            start = time.time()
            # CALL TO CHATBOT, which triggers the RAG pipeline
            result = st.session_state.chatbot.query(user_input)
            elapsed = time.time() - start
        
        # Add bot response
        st.session_state.messages.append({
            'role': 'assistant',
            'content': result['answer'],
            'timestamp': datetime.now().isoformat(),
            'sources': result['sources'],
            'latency_ms': elapsed * 1000
        })
        
        # Display
        st.write(f"**Bot:** {result['answer']}")
        
        if result['sources']:
            st.subheader("📚 Sources")
            for s in result['sources']:
                st.write(f"- {s['title']} ({s['relevance']})")
        
        st.rerun()

with col2:
    st.subheader("📈 Metrics")
    
    if st.session_state.messages:
        user_messages = [m for m in st.session_state.messages if m['role'] == 'user']
        bot_messages = [m for m in st.session_state.messages if m['role'] == 'assistant']
        
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Turns", len(user_messages))
        
        # Latencies
        latencies = [m.get('latency_ms', 0) for m in bot_messages if m.get('latency_ms')]
        if latencies:
            st.metric("Avg Response Time", f"{sum(latencies)/len(latencies):.0f}ms")
        
        # Response lengths
        avg_length = sum(len(m['content']) for m in bot_messages) / len(bot_messages) if bot_messages else 0
        st.metric("Avg Response Length", f"{avg_length:.0f} chars")