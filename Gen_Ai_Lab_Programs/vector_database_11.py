import faiss
from sentence_transformers import SentenceTransformer

documents = [
    "Python is a programming language used for software development.",
    "Computer networks allow devices to communicate and share information.",
    "SQL is used to store and retrieve data from relational databases.",
    "Machine learning enables computers to learn from data.",
    "Artificial intelligence helps machines perform intelligent tasks.",
    "Civil engineering involves designing buildings, roads and bridges."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

query = input("Enter your query: ")

query_embedding = model.encode([query])

distances, indices = index.search(query_embedding, 3)

print("\nMost Relevant Documents:\n")

for i, index_value in enumerate(indices[0]):
    print(f"{i + 1}. {documents[index_value]}")
    print(f"Distance: {distances[0][i]:.4f}")
    print()