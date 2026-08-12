from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Python is a programming language used for AI and data science.",
    "Machine learning allows computers to learn from data.",
    "Deep learning uses artificial neural networks.",
    "Natural Language Processing deals with human language."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))

print("AI Domain Chatbot")
print("Type exit to stop.")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    query_embedding = model.encode([question])

    distance, result = index.search(
        np.array(query_embedding, dtype="float32"), 1
    )

    answer = documents[result[0][0]]

    print("Bot:", answer)