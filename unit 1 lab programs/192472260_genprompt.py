from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

result = generator("The future of AI", max_length=60)

print(result[0]["generated_text"])