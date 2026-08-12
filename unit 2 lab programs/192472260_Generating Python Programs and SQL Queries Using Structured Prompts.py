from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

python_prompt = """Generate a Python program to calculate total sales.
Input: sales = [100, 200, 300]
Output: total sales"""

sql_prompt = """Generate an SQL query.
Table: sales
Columns: product, quantity, price
Task: find total sales for each product"""

python_result = generator(
    python_prompt,
    max_length=100,
    num_return_sequences=1
)

sql_result = generator(
    sql_prompt,
    max_length=100,
    num_return_sequences=1
)

print("PYTHON PROGRAM:")
print(python_result[0]["generated_text"])

print("\nSQL QUERY:")
print(sql_result[0]["generated_text"])