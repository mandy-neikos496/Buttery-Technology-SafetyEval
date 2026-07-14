from openai import OpenAI
import os
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

response = client.chat.completions.create(
    model="nvidia/nemotron-mini-4b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Answer with one word only: which character is from AKEndfield, given that Kyryll Chudomivorich Flins is from Genshin Impact, Camille is from AKEndfield, and Lohen is from Genshin Impact."
        }
    ],
    temperature=0,
    max_tokens=8,
)

print(response.choices[0].message.content)
