from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning"
]

for prompt in prompts:
    print("Prompt:", prompt)
    result = generator(prompt, max_length=40, num_return_sequences=1)
    print(result[0]["generated_text"])
    print("-" * 50)