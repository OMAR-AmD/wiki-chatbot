import streamlit as st
from api_client import WikiChatbotAPIClient
import time

st.set_page_config(page_title="Wiki Chatbot", page_icon="🤖", layout="wide")

# Initialize API client
if 'api_client' not in st.session_state:
    st.session_state.api_client = WikiChatbotAPIClient()

if 'messages' not in st.session_state:
    st.session_state.messages = []

api = st.session_state.api_client

# Check API health
try:
    health = api.health_check()
except:
    st.error("❌ API server not running. Start it with: python api.py")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🤖 Wiki Chatbot")

    if st.button("🔄 New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("History")
    sessions = api.list_sessions()
    if sessions:
        st.write(f"📊 {len(sessions)} conversations")

# Main interface
st.title("💬 Wiki Chatbot")

# Display messages
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")

# Input + Send button
user_input = st.text_input("You:", placeholder="Ask me anything...", key="user_input")
send_button = st.button("Send")

# Process when Send is clicked
if send_button and user_input.strip():
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    with st.spinner("Thinking..."):
        result = api.query(user_input)

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
            st.write(f"- {s['title']} ({s['relevance']})")

    col1, col2, col3 = st.columns(3)
    col1.metric("Time", f"{result['latency_ms']:.0f} ms")
    col2.metric("Sources", len(result['sources']))
    col3.metric("Turn", len([m for m in st.session_state.messages if m['role'] == 'user']))

    # Clear input safely
    st.session_state.pop("user_input", None)
    st.rerun()
