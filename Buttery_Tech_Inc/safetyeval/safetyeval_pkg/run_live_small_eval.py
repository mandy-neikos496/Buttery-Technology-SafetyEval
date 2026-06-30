import json
from datetime import datetime

from safetyeval.data_structures import EthicsAxis
from safetyeval.ingestion import load_items_from_file
from safetyeval.prompts import format_prompt
from safetyeval.evaluation import classify_response, desired_outcomes
from safetyeval.nvidia_client import call_nvidia

items = load_items_from_file("data/sample-data.json")

selected_items = []
for axis in [
    EthicsAxis.BIAS,
    EthicsAxis.HARM,
    EthicsAxis.FACTUALITY,
    EthicsAxis.STEREOTYPE,
]:
    for item in items:
        if item.axis == axis:
            selected_items.append(item)
            break

logs = []

for item in selected_items:
    formatted = format_prompt(item)

    if isinstance(formatted, tuple):
        prompt, letter_to_type = formatted
    else:
        prompt = formatted
        letter_to_type = None
    
    api_result = call_nvidia(prompt)
    raw_text = api_result["raw_text"]

    outcome = classify_response(item, raw_text, letter_to_type)
    is_desired = outcome == desired_outcomes.get(item.axis)

    # Do not display or log raw harm-axis responses in public/shareable outputs.
    safe_raw_response = "[hidden for safety]" if item.axis == EthicsAxis.HARM else raw_text 

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "item_id": item.id,
        "benchmark": item.benchmark,
        "axis": item.axis.value,
        "model_id": api_result["model_id"],
        "raw_response": safe_raw_response,
        "parsed_outcome": outcome.value,
        "is_desired_behavior": is_desired,
        "latency_ms": api_result["latency_ms"],
        "error": None,
    }

    logs.append(log_entry)

    print("\n---")
    print("Item:", item.id)
    print("Axis:", item.axis.value)
    print("Raw response:", safe_raw_response)
    print("Outcome:", outcome.value)
    print("Desired:", is_desired)

with open("live_api_logs.jsonl", "a") as f:
    for entry in logs:
        f.write(json.dumps(entry) + "\n")

print("\nSaved logs to live_api_logs.jsonl")