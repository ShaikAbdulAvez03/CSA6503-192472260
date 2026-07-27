from transformers import pipeline

pipe = pipeline("sentiment-analysis")

print(pipe("Python is easy to learn."))