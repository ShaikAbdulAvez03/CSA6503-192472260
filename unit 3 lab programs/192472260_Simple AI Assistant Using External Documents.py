from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(r"C:\Users\LENOVO\OneDrive\Desktop\CSA6503\unit 3 lab programs\knowledge.txt", "r") as file:
    text = file.read()

documents = [
    line.strip()
    for line in text.split(".")
    if line.strip()
]

embeddings = model.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(
    np.array(embeddings, dtype="float32")
)

print("Document AI Assistant")
print("Ask questions about the document.")
print("Type exit to stop.")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Assistant: Goodbye!")
        break

    query_embedding = model.encode([question])

    distance, result = index.search(
        np.array(query_embedding, dtype="float32"), 1
    )

    answer = documents[result[0][0]]

    print("Assistant:", answer)