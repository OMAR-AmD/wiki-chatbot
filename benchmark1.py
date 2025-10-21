import ollama
import time

# Texte de test : une question qu'on poserait au chatbot
test_questions = [
    "Comment installer les dépendances du projet?",
    "Qu'est-ce que l'architecture microservices?",
    "Quel est le port du serveur de production?"
]

models = ['llama2', 'mistral']
results = {}

for model in models:
    print(f"\n{'='*50}")
    print(f"Testing {model}")
    print(f"{'='*50}")
    
    results[model] = {
        'latencies': [],
        'responses': []
    }
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        start_time = time.time()
        
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': f"Réponds brièvement à cette question:\n{question}"
            }
        ])
        
        latency = time.time() - start_time
        answer = response['message']['content']
        
        results[model]['latencies'].append(latency)
        results[model]['responses'].append(answer)
        
        print(f"Latency: {latency:.2f}s")
        print(f"Answer: {answer[:100]}...")

# Résumé
print(f"\n{'='*50}")
print("RÉSUMÉ BENCHMARK")
print(f"{'='*50}")

for model in models:
    avg_latency = sum(results[model]['latencies']) / len(results[model]['latencies'])
    print(f"\n{model}:")
    print(f"  Latence moyenne: {avg_latency:.2f}s")
    print(f"  Min: {min(results[model]['latencies']):.2f}s")
    print(f"  Max: {max(results[model]['latencies']):.2f}s")