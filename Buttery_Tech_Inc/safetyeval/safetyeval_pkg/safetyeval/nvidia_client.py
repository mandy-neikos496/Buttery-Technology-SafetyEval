import os
import time
import random
from openai import OpenAI

TEMPERATURE = 0.0001
MAX_RETRIES = 5 # Increased from 3 to give rate limits chance to cool down
BASE_RETRY_DELAY = 2

TRANSIENT_ERROR_HINTS = ["504", "503", "429", "timeout"]

def _looks_transient(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in TRANSIENT_ERROR_HINTS)

# Fallback model if no model is specified
def call_nvidia(prompt: str, model_id: str = "nvidia/nemotron-mini-4b-instruct") -> dict: 
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )

    started = time.perf_counter()
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=16
            )

            latency_ms = int((time.perf_counter() - started) * 1000)

            content = response.choices[0].message.content
            raw_text = content.strip() if content else ""

            return {
                "model_id": model_id,
                "raw_text": raw_text,
                "latency_ms": latency_ms
            }
        
        except Exception as exc:
            last_exception = exc
            if not _looks_transient(exc):
                raise
            
            if attempt == MAX_RETRIES:
                break

            sleep_duration = (BASE_RETRY_DELAY ** attempt) + random.uniform(0.5, 1.5)

            is_429 = (getattr(exc, "status_code", None) == 429) or ("429" in str(exc))
            if is_429:
                sleep_duration +=10
                
            print(f" (transient error on {model_id}, retry {attempt}/{MAX_RETRIES - 1}."
                  f"Sleeping {sleep_duration:.2f}s...Error: {exc})")
            time.sleep(sleep_duration)

    raise last_exception