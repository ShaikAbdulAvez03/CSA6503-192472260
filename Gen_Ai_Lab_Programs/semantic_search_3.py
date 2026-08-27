from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Python programming uses variables, functions, loops and classes.",
    "Computer networks allow devices to communicate using protocols such as TCP/IP.",
    "Database management systems store, organize and retrieve structured data.",
    "Artificial intelligence uses machine learning algorithms to solve complex problems.",
    "Civil engineering deals with the design and construction of buildings and bridges."
]

query = input("Enter your search query: ")

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

results = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)

print("\nSemantic Search Results:\n")

for i, (document, score) in enumerate(results, 1):
    print(f"{i}. {document}")
    print(f"   Similarity Score: {score:.4f}\n")