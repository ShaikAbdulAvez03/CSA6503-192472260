import ollama

reference = """
The Eiffel Tower is located in Paris, France.
It was completed in 1889.
It is approximately 330 meters tall including antennas.
"""

question = input("Enter a question: ")

prompt = f"""
Answer the question using only the reference information.

Reference:
{reference}

Question:
{question}

If the answer is not available in the reference, say:
"Information not available in the reference."
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

print("\nReference Information:")
print(reference)

print("\nLLM Answer:")
print(response["message"]["content"])