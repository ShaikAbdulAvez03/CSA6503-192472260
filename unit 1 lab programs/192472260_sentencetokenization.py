import nltk
from nltk.tokenize import sent_tokenize

text = "Python is easy. It is used for AI. It is popular."

sentences = sent_tokenize(text)
print(sentences)