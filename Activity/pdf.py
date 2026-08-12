from pypdf import PdfReader
import chromadb
from openai import OpenAI

PDF_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\CSA6503\Activity\CO2 AT1 Prompting Q.pdf"
DB_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\CSA6503\Activity\chroma_db"
COLLECTION_NAME = "pdfs"


def load_pdf(path):
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


text = load_pdf(PDF_PATH)
print(f"loaded {len(text)} characters")

chunks = chunk_text(text)
print(f"created {len(chunks)} chunks")

client = OpenAI(api_key="YOUR_API_KEY_HERE")
col = client.get_or_create_collection(COLLECTION_NAME)

if col.count() == 0:
    col.add(
        documents=chunks,
        metadatas=[{"source": PDF_PATH, "chunk": i} for i in range(len(chunks))],
        ids=[str(i) for i in range(len(chunks))],
    )
    print(f"vector DB has {col.count()} chunks")


def retrieve(query, k=3):
    results = col.query(query_texts=[query], n_results=k)
    return results["documents"][0], results["metadatas"][0]


PROMPT = """What is the main topic of the document?
Answer ONLY from the context below.
If the answer is not contained within the context, say "I don't know".
Cite the chunk number after each fact, like [c4].

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(query):
    docs, metadata = retrieve(query)
    context = "\n".join([f"[c{m['chunk'] + 1}] {d}" for d, m in zip(docs, metadata)])
    return PROMPT.format(context=context, question=query)


llm = OpenAI()


def answer_query(query):
    prompt = build_prompt(query)
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Enter your query (or 'exit' to quit):")
    while True:
        q = input("Query: ")
        if q.lower() == "exit":
            break
        print("Answer:", answer_query(q))