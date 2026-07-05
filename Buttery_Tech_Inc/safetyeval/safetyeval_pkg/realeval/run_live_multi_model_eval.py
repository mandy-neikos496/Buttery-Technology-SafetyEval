import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from safetyeval.prompts import format_prompt
from safetyeval.evaluation import classify_response, desired_outcomes
from safetyeval.nvidia_client import call_nvidia
from safetyeval.data_structures import EthicsAxis
from safetyeval.real_data import (
    load_real_bbq_items,
    load_real_harmbench_items,
    load_real_truthfulqa_items,
    load_real_stereoset_items,
)

# Models to evaluate
MODELS = [
  # "nvidia/nemotron-mini-4b-instruct", # Smaller model for testing
    "meta/llama-3.3-70b-instruct",
    "google/gemma-2-2b-it",
    "mistralai/mistral-large-3-675b-instruct-2512",
  # "meta/llama-3.2-1b-instruct", # Note: re-add later; fix the error 504 problem
  # "qwen/qwen3.5-122b-a10b", # model always returned "Error code:504"
]

# How many items to pull per axis.
ITEMS_PER_AXIS = 3

# Creates a unique filename for each individual execution session (for data organization)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_LOG = f"run_{TIMESTAMP}.jsonl"

def select_items():
    """Pulls a small real sample from each of the four axes."""
    items  = []
    items += load_real_bbq_items(limit_per_category=ITEMS_PER_AXIS) # bias
    items += load_real_harmbench_items(limit=ITEMS_PER_AXIS) # harm
    items += load_real_truthfulqa_items(limit=ITEMS_PER_AXIS) # factuality
    items += load_real_stereoset_items(limit=ITEMS_PER_AXIS) # stereotype
    return items

# Run requests in parallel for efficiency
def process_item(item, model_id):
    """Worker function to process a single item for a specific"""
    formatted = format_prompt(item)
    if isinstance(formatted, tuple):
        prompt, letter_to_type = formatted
    else:
        prompt = formatted 
        letter_to_type = None

    try:
        print(f" Calling {model_id} on {item.id}...", flush=True)
        api_result = call_nvidia(prompt, model_id=model_id)
        raw_text = api_result["raw_text"]

        outcome = classify_response(item, raw_text, letter_to_type)
        is_desired = outcome == desired_outcomes.get(item.axis)

        # Never log raw harm-axis responses
        safe_raw_response = (
            "[hidden for safety]" if item.axis == EthicsAxis.HARM else raw_text
        )

        print(f" -> {model_id} finished {item.id}: {outcome.value}")

        return {
            "timestamp": datetime.now().isoformat(),
            "item_id": item.id,
            "benchmark": item.benchmark,
            "axis": item.axis.value,
            "model_id": model_id,
            "raw_response": safe_raw_response,
            "parsed_outcome": outcome.value,
            "is_desired_behavior": is_desired,
            "latency_ms": api_result["latency_ms"],
            "error": None,
        }
    
    except Exception as exc:
        print(f" -> ERROR on {model_id} / {item.id}: {exc}")
        return {
            "timestamp": datetime.now().isoformat(),
            "item_id": item.id,
            "benchmark": item.benchmark,
            "axis": item.axis.value,
            "model_id": model_id,
            "raw_response": None,
            "parsed_outcome": "error",
            "is_desired_behavior": False,
            "latency_ms": None,
            "error": str(exc),
        }

def main():
    items = select_items()

    # List of all combinations of tasks to run
    tasks = [(item, model_id) for model_id in MODELS for item in items]

    print(f"Starting evaluation of {len(tasks)} total combinations using parallel threads...")

    completed = 0

    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_item, item, model_id) for item, model_id in tasks]

            for future in as_completed(futures):
                result = future.result()

                f.write(json.dumps(result) + "\n")
                f.flush()

                completed += 1
                print(f"[{completed}/{len(tasks)} saved]")

    print(f"\nSaved {completed} log entries to {OUTPUT_LOG}")

if __name__ == "__main__":
    main()