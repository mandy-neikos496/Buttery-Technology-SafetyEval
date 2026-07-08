import os
import glob
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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

VALID_MODELS = [
        "meta/llama-3.3-70b-instruct",
        "google/gemma-2-2b-it",
        "meta/llama-3.2-1b-instruct",
        "qwen/qwen3.5-122b-a10b",
]

def build_items_by_id():
    items = []
    items += load_real_bbq_items(limit_per_category=ITEMS_PER_AXIS)
    items += load_real_harmbench_items(limit=ITEMS_PER_AXIS)
    items += load_real_truthfulqa_items(limit=ITEMS_PER_AXIS)
    items += load_real_stereoset_items(limit=ITEMS_PER_AXIS)
    return {item.id: item for item in items}

def main():
    log_files = glob.glob("run_*.jsonl")
    if not log_files:
        print("No run_*.jsonl files found.")
        return
    
    print(f"Found {len(log_files)} run files. Analyzing for most recent items...")
    items_by_id = build_items_by_id()

    raw_rows = []
    for file_path in log_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_rows.append(json.loads(line))

    raw_rows.sort(key=lambda r: r.get("timestamp", ""))

    most_recent_dataset = {}
    for row in raw_rows:
        model_id = row.get("model_id")
        if model_id not in VALID_MODELS or row.get("parsed_outcome") == "error":
            continue

        key = (model_id, row["item_id"])
        most_recent_dataset[key] = row

    all_recent_rows = list(most_recent_dataset.values())
    harm_rows = [r for r in all_recent_rows if r.get("axis") == "harm"]

    print(f"\nDe-duplicated dataset contains {len(all_recent_rows)} total active items.")
    print(f"Found exactly {len(harm_rows)} unique, most-recent harm items to process.")

    processed_rows = []
    current_harm_idx = 0

    # Loop through rows and evaluate active harm data live
    for row in all_recent_rows:
        if row.get("axis") != "harm":
            processed_rows.append(row)
            continue

        current_harm_idx += 1
        model_id = row["model_id"]
        item_id = row["item_id"]
        item = items_by_id.get(item_id)

        if item is None:
            print(f" -> [{current_harm_idx}/ {len(harm_rows)}] Warning: Item {item_id} not found in source datasets.")
            processed_rows.append(row)
            continue

        print(f" -> [{current_harm_idx}/{len(harm_rows)}] Running dual-evaluation for {model_id} on item {item_id}...")

        time.sleep(20.0)

        formatted = format_prompt(item)
        prompt, letter_to_type = formatted if isinstance(formatted, tuple) else (formatted, None)

        try:
            # Query live model text response
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_nvidia, prompt, model_id=model_id)
                api_result = call_nvidia(prompt, model_id=model_id)

            raw_text = api_result["raw_text"]

            # Compute standard regex evaluation
            outcome = classify_response(item, raw_text, letter_to_type)

            # Compute LLM Judge evaluation on the fresh text block
            judge_outcome = None
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    judge_future = executor.submit(classify_refusal_llm_judge, item, raw_text)   
                    judge_outcome_obj = judge_future.result(timeout=60.0)
                judge_outcome = judge_outcome_obj.value if judge_outcome_obj else None
            except TimeoutError:
                print(f" -> Judge evaluation timed out (60s limit reached).")
                judge_outcome = "error"
            except Exception as j_err:
                print(f" -> Judge evaluation tracking error: {j_err}")
                judge_outcome = "error"

            updated_row = {
                "timestamp": datetime.now().isoformat(), 
                "item_id": item.id,
                "benchmark": item.benchmark,
                "axis": item.axis.value,
                "model_id": model_id,
                "raw_response": "[hidden for safety]",
                "parsed_outcome": outcome.value,
                "llm_judge_outcome": judge_outcome,
                "is_desired_behavior": outcome == desired_outcomes.get(item.axis), 
                "latency_ms": api_result["latency_ms"],
                "error": None,
            }
            processed_rows.append(updated_row)
            print(f" Success: Regex={outcome.value} | Judge={judge_outcome}")

        except Exception as exc:
            print(f" -> API execution failure on {model_id}: {exc}")
            # Fall back to original logged record if network problem occurs
            processed_rows.append(row)
        except Exception as exc:
            print(f" -> API execution failure on {model_id}: {exc}")
            processed_rows.append(row)

    # Save clean & updated data structure
    output_file = "run_final_consolidated.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for row in processed_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nCompleted! Saved master clean file at: {output_file}")
    print("Now point cell #1 in analysis.ipynb to load this file.")

if __name__ == "__main__":
    main()