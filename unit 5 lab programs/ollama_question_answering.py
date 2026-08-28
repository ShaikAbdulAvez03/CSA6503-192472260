import ollama

context = """
Artificial Intelligence is a technology that allows computers to perform
tasks that normally require human intelligence. AI is used in healthcare,
education, banking, transportation and many other fields.
"""

question = input("Enter your question: ")

prompt = f"""
Answer the question using the information given below.

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAnswer:")
print(response["message"]["content"])