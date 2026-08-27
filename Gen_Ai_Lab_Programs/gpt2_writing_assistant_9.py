from transformers import pipeline

writer = pipeline(
    "text-generation",
    model="gpt2"
)

prompt = """
Engineering students are developing a smart campus system that uses
artificial intelligence to improve energy efficiency.
"""

result = writer(
    prompt,
    max_new_tokens=100,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7
)

print("Engineering Writing Assistant\n")
print("Prompt:")
print(prompt)

print("\nGenerated Output:")
print(result[0]["generated_text"])