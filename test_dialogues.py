from chatbot import WikiChatbot
from chat_storage import ChatStorage
import json
import time

# Dialogues réalistes
test_dialogues = [
    {
        'name': 'Dialogue 1: Onboarding Complet',
        'turns': [
            "Hi, I'm new here. How do I get started?",
            "Okay, and what's required?",
            "How long does setup usually take?",
            "What if I run into errors?"
        ]
    },
    {
        'name': 'Dialogue 2: Architecture Questions',
        'turns': [
            "Can you explain the system architecture?",
            "Which components are most critical?",
            "How do they communicate?",
            "What database do we use?",
            "Is it scalable?"
        ]
    },
    {
        'name': 'Dialogue 3: Troubleshooting',
        'turns': [
            "I'm getting connection errors",
            "How do I check the service status?",
            "What should I look for in logs?",
            "Is there a diagnostic tool?"
        ]
    }
]

print("="*70)
print("DIALOGUE TEST SUITE")
print("="*70)

storage = ChatStorage()
results = {
    'dialogues': [],
    'summary': {}
}

total_latency = 0
total_turns = 0

for dialogue in test_dialogues:
    print(f"\n[{dialogue['name']}]")
    print("-"*70)
    
    chatbot = WikiChatbot()
    dialogue_results = {
        'name': dialogue['name'],
        'turns': [],
        'avg_latency_ms': 0
    }
    
    dialogue_latencies = []
    
    for i, user_input in enumerate(dialogue['turns'], 1):
        print(f"Turn {i}: {user_input[:50]}...")
        
        start = time.time()
        result = chatbot.query(user_input)
        latency = (time.time() - start) * 1000
        
        dialogue_latencies.append(latency)
        total_latency += latency
        total_turns += 1
        
        dialogue_results['turns'].append({
            'user': user_input,
            'bot_response': result['answer'][:100],
            'latency_ms': latency,
            'sources': len(result['sources']),
            'success': result['success']
        })
        
        print(f"  ✓ {latency:.0f}ms, {len(result['sources'])} sources")
    
    dialogue_results['avg_latency_ms'] = sum(dialogue_latencies) / len(dialogue_latencies)
    
    # Sauvegarder conversation
    session_id = f"dialogue_{len(results['dialogues'])+1}"
    storage.save(session_id, chatbot.get_conversation())
    
    results['dialogues'].append(dialogue_results)

# Résumé
results['summary'] = {
    'total_dialogues': len(test_dialogues),
    'total_turns': total_turns,
    'avg_latency_ms': total_latency / total_turns if total_turns > 0 else 0,
    'total_time_seconds': total_latency / 1000
}

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total dialogues: {results['summary']['total_dialogues']}")
print(f"Total turns: {results['summary']['total_turns']}")
print(f"Average latency: {results['summary']['avg_latency_ms']:.0f}ms")
print(f"Total time: {results['summary']['total_time_seconds']:.1f}s")

# Sauvegarder résultats
with open('JOUR4_DIALOGUE_TESTS.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to JOUR4_DIALOGUE_TESTS.json")