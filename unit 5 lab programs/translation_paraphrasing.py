import ollama

text = input("Enter text: ")

print("\n1. Translation")
print("2. Paraphrasing")

choice = input("Enter choice: ")

if choice == "1":
    language = input("Enter target language: ")

    prompt = f"Translate the following text into {language}:\n{text}"

elif choice == "2":
    prompt = f"Paraphrase the following text without changing its meaning:\n{text}"

else:
    print("Invalid choice")
    exit()

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nResult:")
print(response["message"]["content"])