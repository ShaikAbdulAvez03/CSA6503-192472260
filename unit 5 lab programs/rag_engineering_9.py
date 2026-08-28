import ollama
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="engineering_docs"
)

with open("engineering_docs/motor.txt", "r") as file:
    document = file.read()

chunks = [
    document[i:i + 500]
    for i in range(0, len(document), 500)
]

embeddings = model.encode(chunks).tolist()

collection.upsert(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

question = input("Enter technical question: ")

query_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

context = "\n".join(results["documents"][0])

prompt = f"""
Answer the technical question using only the provided context.

Context:
{context}

Question:
{question}

Give a clear and accurate answer.
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nRetrieved Information:")
print(context)

print("\nAnswer:")
print(response["message"]["content"])