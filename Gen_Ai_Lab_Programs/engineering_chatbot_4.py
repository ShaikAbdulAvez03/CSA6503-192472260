from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "What is Python?",
    "How do I use a for loop in Python?",
    "What is a computer network?",
    "What is an IP address?",
    "What is a database?",
    "What is SQL?",
    "What is artificial intelligence?",
    "What is machine learning?",
    "What is civil engineering?"
]

answers = [
    "Python is a high-level programming language used for web development, automation, data science and artificial intelligence.",
    "A for loop is used to repeatedly execute a block of code for each item in a sequence.",
    "A computer network is a group of connected devices that communicate and share resources.",
    "An IP address is a unique address used to identify a device on a network.",
    "A database is an organized collection of data that can be stored, managed and retrieved.",
    "SQL is a language used to create, read, update and manage data in relational databases.",
    "Artificial intelligence is a field of computing that enables machines to perform tasks requiring human-like intelligence.",
    "Machine learning is a branch of AI where systems learn patterns from data and make predictions.",
    "Civil engineering involves designing, constructing and maintaining infrastructure such as buildings, roads and bridges."
]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

print("Engineering Support Chatbot")
print("Type 'exit' to stop.\n")

while True:
    user_question = input("Student: ")

    if user_question.lower() == "exit":
        print("Chatbot: Thank you!")
        break

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )[0]

    best_match = similarity.argmax()

    if similarity[best_match] < 0.2:
        print("Chatbot: Sorry, I don't have a relevant solution for that question.")
    else:
        print("Chatbot:", answers[best_match])