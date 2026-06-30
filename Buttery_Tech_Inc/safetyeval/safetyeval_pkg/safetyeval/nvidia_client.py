import os
import time
from openai import OpenAI

def call_nvidia(prompt: str, model_id: str = "nvidia/nemotron-mini-4b-instruct") -> dict: 
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )

    started = time.perf_counter()

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16
    )

    latency_ms = int((time.perf_counter() - started) * 1000)

    return {
        "model_id": model_id,
        "raw_text": response.choices[0].message.content.strip(),
        "latency_ms": latency_ms
    }