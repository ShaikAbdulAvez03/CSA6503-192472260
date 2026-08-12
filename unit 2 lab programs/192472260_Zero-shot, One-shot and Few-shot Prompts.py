from transformers import pipeline

model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "My order has not arrived yet."

labels = ["complaint", "question", "feedback"]

zero_shot = model(text, labels)

one_shot = model(
    "Example: My payment failed = complaint\n" + text,
    labels
)

few_shot = model(
    "Examples:\nMy payment failed = complaint\n"
    "What is my order status? = question\n"
    "The service was excellent = feedback\n" + text,
    labels
)

print("Zero-shot:")
print(zero_shot)

print("\nOne-shot:")
print(one_shot)

print("\nFew-shot:")
print(few_shot)