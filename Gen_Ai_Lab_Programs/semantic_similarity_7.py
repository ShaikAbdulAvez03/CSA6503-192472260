from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Python is a programming language used to develop software applications.",
    "Computer networks connect computers and allow them to exchange information.",
    "SQL is used to store, retrieve and manage data in relational databases.",
    "Machine learning allows computers to learn patterns from data.",
    "Civil engineering involves designing buildings, roads and bridges.",
    "Cybersecurity protects computer systems and networks from attacks."
]

query = input("Enter your query: ")

model = SentenceTransformer("all-MiniLM-L6-v2")

document_embeddings = model.encode(documents)
query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

results = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nMost Relevant Documents:\n")

for i, (document, score) in enumerate(results, 1):
    print(f"{i}. {document}")
    print(f"Similarity Score: {score:.4f}")
    print()