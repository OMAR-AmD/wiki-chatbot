from chatbot import WikiChatbot
import json

print("="*70)
print("MULTI-TURN CONVERSATION TESTS")
print("="*70)

# Test 1 : Conversation linéaire
print("\n[Test 1] Linear Conversation")
print("-"*70)

chatbot = WikiChatbot()

conversation_1 = [
    "How do I setup the project?",
    "What do I need to install?",
    "How do I run it after setup?"
]

for query in conversation_1:
    print(f"\nUser: {query}")
    result = chatbot.query(query)
    print(f"Bot: {result['answer'][:200]}...")
    print(f"Sources: {len(result['sources'])} found")

# Sauvegarder
chatbot.save_conversation("conversations/test1_linear.json")

# Test 2 : Conversation avec suivi
print("\n\n[Test 2] Conversation with Context Switching")
print("-"*70)

chatbot2 = WikiChatbot()

conversation_2 = [
    "What's the system architecture?",
    "Which database do we use?",
    "How can I connect to it?",
    "What are the credentials?",
    "How do I debug connection errors?"
]

for query in conversation_2:
    print(f"\nUser: {query}")
    result = chatbot2.query(query)
    print(f"Bot: {result['answer'][:200]}...")
    print(f"Turn: {result['turn']}")

# Sauvegarder
chatbot2.save_conversation("conversations/test2_context.json")

# Test 3 : Analyser patterns
print("\n\n[Test 3] Conversation Analytics")
print("-"*70)

def analyze_conversation(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    
    print(f"Session: {data['session_id']}")
    print(f"Total turns: {len([m for m in history if m['role'] == 'user'])}")
    print(f"Average response length: {sum(len(m['content']) for m in history if m['role'] == 'assistant') // max(1, len([m for m in history if m['role'] == 'assistant']))} chars")
    
    print("\nConversation flow:")
    for msg in history:
        content_preview = msg['content'][:60].replace('\n', ' ')
        print(f"  {msg['role'].upper()}: {content_preview}...")

analyze_conversation("conversations/test1_linear.json")
analyze_conversation("conversations/test2_context.json")

print("\n✅ Multi-turn tests complete!")