from openai import OpenAI
from dotenv import load_dotenv
from memory.client import mem

load_dotenv()

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' ten times fast.",
        },
    ],
    stream=True,
)

for event in stream:
    print(event)
