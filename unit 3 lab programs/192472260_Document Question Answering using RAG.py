from sentence_transformers import SentenceTransformer
import faiss

docs = ["Python is used for programming.",
        "Machine learning is a part of artificial intelligence.",
        "RAG combines search with text generation."]

model = SentenceTransformer("all-MiniLM-L6-v2")
v = model.encode(docs)
index = faiss.IndexFlatL2(v.shape[1])
index.add(v)

q = input("Question: ")
x = model.encode([q])
d, i = index.search(x, 1)

print("Answer:", docs[i[0][0]])