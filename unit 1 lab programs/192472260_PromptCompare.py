from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompts = [
    "Artificial Intelligence",
    "Machine Learning",
    "Data Science"
]

for prompt in prompts:
    print("Prompt:", prompt)
    output = generator(prompt, max_length=40)
    print(output[0]["generated_text"])
    print("-" * 50)