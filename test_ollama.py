import ollama

# Test simple
response = ollama.chat(model='llama2', messages=[
    {
        'role': 'user',
        'content': 'Dis moi un joke court sur Python'
    }
])

print(response['message']['content'])