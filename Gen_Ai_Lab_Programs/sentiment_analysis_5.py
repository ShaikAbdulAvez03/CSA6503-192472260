from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

feedback = [
    "The placement training was excellent and very useful.",
    "The placement process was confusing and disappointing.",
    "The placement cell conducted the session yesterday."
]

print("Student Feedback Sentiment Analysis\n")

for text in feedback:
    result = sentiment(text)[0]

    print("Feedback:", text)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))
    print()