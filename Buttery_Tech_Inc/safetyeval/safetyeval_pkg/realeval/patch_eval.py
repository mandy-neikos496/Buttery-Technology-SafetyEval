import os
import glob
import json
import time
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from safetyeval.prompts import format_prompt
from safetyeval.evaluation import classify_response, desired_outcomes
from safetyeval.classifiers import classify_refusal_llm_judge
from safetyeval.nvidia_client import call_nvidia
from safetyeval.data_structures import EthicsAxis
from safetyeval.real_data import (
    load_real_bbq_items,
    load_real_harmbench_items,
    load_real_truthfulqa_items,
    load_real_stereoset_items,
)

# Note: match ITEMS_PER_AXIS used in run_live_multi_model_eval.py
ITEMS_PER_AXIS = 10

def get_latest_log_file():
    log_files = glob.glob("run_*.jsonl")
    if not log_files:
        raise FileNotFoundError("No run_*.jsonl files found in this folder.")
    return max(log_files, key=os.path.getmtime)

def build_items_by_id():
    """Rebuilds same items the original run used."""
    items = []
    items += load_real_bbq_items(limit_per_category=ITEMS_PER_AXIS)
    items += load_real_harmbench_items(limit=ITEMS_PER_AXIS)
    items += load_real_truthfulqa_items(limit=ITEMS_PER_AXIS)
    items += load_real_stereoset_items(limit=ITEMS_PER_AXIS)
    return {item.id: item for item in items}

def call_and_score(item, model_id):
    """Same shape as process_item() in run_live_multi_model_eval.py."""
    formatted = format_prompt(item)
    prompt, letter_to_type = formatted if isinstance(formatted, tuple) else (formatted, None)

    max_retries = 4
    base_backoff_seconds = 30.0

    for attempt in range(max_retries + 1):
        try: 
            api_result = call_nvidia(prompt, model_id=model_id)
            raw_text = api_result["raw_text"]
            outcome = classify_response(item, raw_text, letter_to_type)

            return {
                "timestamp": datetime.now().isoformat(),
                "item_id": item.id,
                    "benchmark": item.benchmark,
                    "axis": item.axis.value,
                    "model_id": model_id,
                    "raw_response": "[hidden for safety]" if item.axis == EthicsAxis.HARM else raw_text,
                    "parsed_outcome": outcome.value,
                    "is_desired_behavior": outcome == desired_outcomes.get(item.axis),
                    "latency_ms": api_result["latency_ms"],
                    "error": None,
            }
        except Exception as exc:
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
    try:
        log_file = get_latest_log_file()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"Patching: {log_file}")

    # Back up original before changing anything.
    shutil.copy(log_file, log_file + ".bak")
    print(f"Backup saved to {log_file}.bak")

    # Read every row one time.
    with open(log_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    items_by_id = build_items_by_id()

    keep_rows = [] # successes and any error that cannot retry
    retry_tasks = [] 

    for row in rows:
        if row["parsed_outcome"] != "error":
            keep_rows.append(row)
            continue
        
        item = items_by_id.get(row["item_id"])
        if item is None:
            print(f"Could not find item '{row['item_id']}' -> leaving its error row as-is.")
            keep_rows.append(row)
        else:
            retry_tasks.append((item, row["model_id"]))

    if not retry_tasks:
        print("No retryable error rows found -> nothing to patch.")
        return
    
    print(f"Retrying {len(retry_tasks)} failed combination(s), one at a time...")

    new_rows = []
    
    for idx, (item, model_id) in enumerate(retry_tasks):
        print(f"\n[Progress {idx+1}/{len(retry_tasks)}] Preparing request for {model_id}...")

        time.sleep(20.0)

        result = call_and_score(item, model_id)
        new_rows.append(result)
        print(f" Finished: {result['model_id']} / {result['item_id']}: {result['parsed_outcome']}")
    # Replace the file
    with open(log_file, "w", encoding="utf-8") as f:
        for row in keep_rows + new_rows:
            f.write(json.dumps(row) + "\n")

    fixed = sum(1 for r in new_rows if r["parsed_outcome"] != "error")
    print(f"\nDone. {fixed}/{len(retry_tasks)} retried rows now succeeded.")
    print(f"Original file is backed up at {log_file}.bak in case anything looks wrong.")

if __name__ == "__main__":
    main()