import streamlit as st
from api_client import WikiChatbotAPIClient
import time
import requests # Import requests to handle potential exceptions better

st.set_page_config(page_title="Wiki Chatbot", page_icon="🤖", layout="wide")

# --- INITIALIZATION & API CHECK ---

# 1. Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = WikiChatbotAPIClient()
    
api = st.session_state.api_client

# Check API health
try:
    health = api.health_check()
except requests.exceptions.ConnectionError:
    st.error("❌ **API server not running.** Start the backend with: `python api.py`")
    st.stop()

# 2. Initialize or load the current session ID
if 'session_id' not in st.session_state:
    try:
        # Call the backend to create a new session ID
        new_session = api.new_session()
        st.session_state.session_id = new_session['session_id']
    except Exception as e:
        st.error(f"Could not initialize session ID from API: {e}")
        st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 Wiki Chatbot")
    st.caption(f"Current Session: **{st.session_state.session_id}**") # Show current session ID

    # 3. Update 'New Chat' to generate a fresh session ID
    if st.button("🔄 New Chat"):
        st.session_state.messages = []
        new_session = api.new_session() 
        st.session_state.session_id = new_session['session_id']
        st.rerun()

    st.subheader("History")
    sessions = api.list_sessions()
    if sessions:
        st.write(f"📊 {len(sessions)} conversations")

# --- MAIN INTERFACE ---
st.title("💬 Wiki Chatbot")

# Display messages (NO CHANGE)
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")

# Input + Send button (NO CHANGE)
user_input = st.text_input("You:", placeholder="Ask me anything...", key="user_input")
send_button = st.button("Send")

# --- QUERY LOGIC ---

# Process when Send is clicked
if send_button and user_input.strip():
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    with st.spinner("Thinking..."):
        # 4. MODIFIED: Pass the session_id to the API query
        result = api.query(
            query=user_input,
            session_id=st.session_state.session_id # <--- Session ID passed here
        )

    # Add assistant message
    st.session_state.messages.append({
        'role': 'assistant',
        'content': result['answer']
    })

    # Display response
    st.markdown(f"**🤖 Bot:** {result['answer']}")

    if result.get('sources'):
        st.subheader("📚 Sources")
        for s in result['sources']:
            # Assuming the source dict has 'title' and 'relevance' as before
            relevance_display = f" ({s['relevance']})" if s.get('relevance') else ""
            st.write(f"- {s['title']}{relevance_display}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Time", f"{result['latency_ms']:.0f} ms")
    col2.metric("Sources", len(result['sources']))
    col3.metric("Turn", len([m for m in st.session_state.messages if m['role'] == 'user']))

    # Clear input safely
    st.session_state.pop("user_input", None)
    st.rerun()
