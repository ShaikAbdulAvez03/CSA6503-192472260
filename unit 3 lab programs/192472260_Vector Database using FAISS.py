import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = ["Python is a programming language",
        "AI is used in many applications",
        "Football is a popular sport"]

vectors = model.encode(docs)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

q = input("Search: ")
v = model.encode([q])
d, i = index.search(v, 1)

print("Result:", docs[i[0][0]])