from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # ignored
    base_url="http://localhost:8000/v1"
)

resp = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Hello from OpenAI SDK"}]
)

print(resp.choices[0].message.content)
