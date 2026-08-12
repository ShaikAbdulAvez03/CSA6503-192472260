from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

prompt = "Write a short SMS giving farmers an advisory about today's weather."

response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt
)

print(response.output_text)