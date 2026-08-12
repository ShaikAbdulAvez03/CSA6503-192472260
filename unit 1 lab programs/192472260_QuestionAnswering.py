from transformers import pipeline

qa = pipeline("question-answering")

context = "Artificial Intelligence is the simulation of human intelligence by machines."

question = "What is Artificial Intelligence?"

result = qa(question=question, context=context)

print(result["answer"])