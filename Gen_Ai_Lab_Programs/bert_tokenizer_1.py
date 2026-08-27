from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

feedback = [
    "The course was very interesting and useful.",
    "The lectures were difficult to understand.",
    "I really enjoyed the practical sessions.",
    "The assignments were too lengthy and confusing."
]

for sentence in feedback:
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("Feedback :", sentence)
    print("Tokens   :", tokens)
    print("Token IDs:", token_ids)
    print("-" * 60)