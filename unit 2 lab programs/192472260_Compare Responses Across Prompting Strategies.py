from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

prompt1 = "Explain banking services."
prompt2 = "Explain banking services in 3 simple sentences."
prompt3 = "You are a banking FAQ assistant. Answer: How can I reset my password?"

r1 = generator(prompt1, max_length=60, num_return_sequences=1)
r2 = generator(prompt2, max_length=60, num_return_sequences=1)
r3 = generator(prompt3, max_length=60, num_return_sequences=1)

print("PROMPT 1:")
print(r1[0]["generated_text"])

print("\nPROMPT 2:")
print(r2[0]["generated_text"])

print("\nPROMPT 3:")
print(r3[0]["generated_text"])