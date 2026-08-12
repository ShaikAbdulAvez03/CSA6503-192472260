from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

a = input("Enter first text: ")
b = input("Enter second text: ")

x = model.encode(a, convert_to_tensor=True)
y = model.encode(b, convert_to_tensor=True)

print("Similarity:", util.cos_sim(x, y).item())