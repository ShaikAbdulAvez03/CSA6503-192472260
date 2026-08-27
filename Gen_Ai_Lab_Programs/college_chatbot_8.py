from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="distilgpt2"
)

print("Engineering College Chatbot")
print("Type 'exit' to stop.\n")

while True:
    question = input("Student: ")

    if question.lower() == "exit":
        print("Chatbot: Thank you!")
        break

    prompt = f"Engineering college support chatbot. Student question: {question}\nAnswer:"

    result = chatbot(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

    print("\nChatbot:", result[0]["generated_text"])
    print()