from chatbot import WikiChatbot
import json

# Critères d'évaluation manuelle
evaluation_criteria = {
    'relevance': 'Est-ce que la réponse adresse la question?',
    'accuracy': 'Est-ce que la réponse est correcte?',
    'completeness': 'Est-ce que la réponse est complète?',
    'clarity': 'Est-ce que la réponse est claire et bien structurée?',
    'usefulness': 'Est-ce que la réponse est utile pour le user?'
}

test_queries = [
    "How do I setup the project locally?",
    "What is the system architecture?",
    "How do I debug connection errors?",
    "What database do we use?",
    "What are the coding standards?"
]

print("="*70)
print("RESPONSE QUALITY EVALUATION")
print("="*70)

chatbot = WikiChatbot()
evaluations = []

for query in test_queries:
    print(f"\n\nQuery: {query}")
    print("-"*70)
    
    result = chatbot.query(query)
    
    print(f"Response: {result['answer'][:300]}...")
    print(f"\nSources: {len(result['sources'])}")
    for s in result['sources']:
        print(f"  - {s['title']} ({s['relevance']})")
    
    # Évaluation
    print(f"\nManual Evaluation (1-5 scale):")
    
    eval_data = {'query': query, 'scores': {}}
    
    for criterion, description in evaluation_criteria.items():
        print(f"  {criterion}: {description}")
        score = input(f"    Score (1-5): ").strip()
        try:
            eval_data['scores'][criterion] = int(score)
        except:
            eval_data['scores'][criterion] = 3
    
    avg_score = sum(eval_data['scores'].values()) / len(eval_data['scores'])
    eval_data['average_score'] = avg_score
    
    evaluations.append(eval_data)
    
    print(f"\n  Average: {avg_score:.1f}/5.0")

# Résumé
print("\n\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

overall_scores = []
for eval_item in evaluations:
    overall_scores.append(eval_item['average_score'])
    print(f"\nQuery: {eval_item['query'][:50]}...")
    print(f"Average Score: {eval_item['average_score']:.1f}/5.0")

final_average = sum(overall_scores) / len(overall_scores)
print(f"\n\nOverall Quality Score: {final_average:.1f}/5.0")

# Sauvegarder
with open('JOUR4_EVALUATION.json', 'w') as f:
    json.dump({
        'evaluations': evaluations,
        'overall_score': final_average
    }, f, indent=2)

print("✅ Evaluation saved to JOUR4_EVALUATION.json")