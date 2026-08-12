from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """Our company is launching a new smart manufacturing system.
The system improves production speed, reduces errors and helps employees
monitor machines in real time."""

inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

output = model.generate(
    **inputs,
    max_length=50,
    min_length=15,
    num_beams=4
)

summary = tokenizer.decode(output[0], skip_special_tokens=True)

generator = pipeline("text-generation", model="distilgpt2")

email = generator(
    "Write a professional email about a new manufacturing system:",
    max_length=80,
    num_return_sequences=1
)

content = generator(
    "Smart manufacturing improves industries by",
    max_length=60,
    num_return_sequences=1
)

print("SUMMARY:")
print(summary)

print("\nEMAIL:")
print(email[0]["generated_text"])

print("\nCONTENT:")
print(content[0]["generated_text"])