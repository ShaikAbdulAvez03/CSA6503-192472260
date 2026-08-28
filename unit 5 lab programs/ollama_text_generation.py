import ollama

prompt = input("Enter a prompt: ")

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nGenerated Text:")
print(response["message"]["content"])