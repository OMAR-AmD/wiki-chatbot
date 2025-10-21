import json
from rag_retriever import WikiRAGRetriever

# Questions de test avec réponses attendues
test_cases = [
    {
        'query': "How do I setup the project locally?",
        'expected_source': 'setup_local.md',
        'category': 'setup'
    },
    {
        'query': "What is the system architecture?",
        'expected_source': 'architecture.md',
        'category': 'architecture'
    },
    {
        'query': "What database do we use?",
        'expected_source': 'architecture.md',
        'category': 'architecture'
    },
    {
        'query': "How do I fix connection errors?",
        'expected_source': 'troubleshooting.md',
        'category': 'troubleshooting'
    },
    {
        'query': "What are the Python naming conventions?",
        'expected_source': 'coding_standards.md',
        'category': 'standards'
    }
]

print("="*70)
print("RAG EVALUATION")
print("="*70)

retriever = WikiRAGRetriever()

correct = 0
total = len(test_cases)

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}/{total}: {test['query']}")
    print("-"*70)
    
    # Retrieve
    docs = retriever.retrieve(test['query'], top_k=3)
    
    if docs:
        top_source = docs[0]['source']
        expected = test['expected_source']
        relevance = docs[0]['relevance']
        
        is_correct = top_source == expected
        correct += int(is_correct)
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Expected: {expected}")
        print(f"   Got: {top_source} (relevance: {relevance:.0%})")
    else:
        print("❌ No results found")

print("\n" + "="*70)
accuracy = (correct / total) * 100
print(f"Accuracy: {correct}/{total} ({accuracy:.0f}%)")

if accuracy >= 80:
    print("✅ RAG precision acceptable!")
else:
    print("⚠️  RAG needs improvement")

# Sauvegarder résultats
with open('JOUR2_RAG_EVALUATION.json', 'w') as f:
    json.dump({
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'test_cases': test_cases
    }, f, indent=2)