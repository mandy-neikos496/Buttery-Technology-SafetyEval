import argparse

from safetyeval.prompts import format_prompt
from safetyeval.nvidia_client import call_nvidia
from safetyeval.evaluation import classify_response, desired_outcomes
from safetyeval.data_structures import EthicsAxis
from safetyeval.reporting import generate_html_report

from safetyeval.real_data import (
    load_real_bbq_items,
    load_real_harmbench_items,
    load_real_truthfulqa_items,
    load_real_stereoset_items,
)

def build_parser():
    parser = argparse.ArgumentParser(prog="safetyeval")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--axes", required=True)
    run.add_argument("--out", default="report.html")
    run.add_argument("--limit", type=int, default=10)

    return parser

def load_selected_items(axes, limit):
    loaders = {
        "bias": lambda: load_real_bbq_items(
            limit_per_category=limit
        ),
        "harm": lambda: load_real_harmbench_items(
            limit=limit
        ),
        "factuality": lambda: load_real_truthfulqa_items(
            limit=limit
        ),
        "stereotype": lambda: load_real_stereoset_items(
            limit=limit
        ),
    }

    items = []

    for axis in axes:
        axis_items = loaders[axis]()
        items.extend(axis_items)
        print(f"Loaded {len(axis_items)} {axis} item(s)")

    return items

def prepare_prompts(items):
    prepared = []

    for item in items:
        formatted = format_prompt(item)

        if isinstance(formatted, tuple):
            prompt, letter_to_type = formatted
        else:
            prompt = formatted
            letter_to_type = None

        prepared.append((item, prompt, letter_to_type))
        print(f"Prepared prompt for {item.id} ({item.axis.value})")

    return prepared

def evaluate_prepared_items(prepared, model_id):
    results = []

    for number, (item, prompt, letter_to_type) in enumerate(
        prepared,
        start=1,
    ):
        print(
            f"Evaluating {number}/{len(prepared)}: "
            f"{item.id} with {model_id}"
        )

        try:
            api_result = call_nvidia(prompt, model_id=model_id)
            raw_text = api_result["raw_text"]

            outcome = classify_response(
                item,
                raw_text,
                letter_to_type,
            )

            is_desired = outcome == desired_outcomes.get(item.axis)

            # Never log raw HarmBench responses
            safe_response = (
                "[hidden for safety]"
                if item.axis == EthicsAxis.HARM
                else raw_text
            )

            results.append(
                {
                    "item_id": item.id,
                    "benchmark": item.benchmark,
                    "axis": item.axis.value,
                    "model_id": model_id,
                    "raw_response": safe_response,
                    "parsed_outcome": outcome.value,
                    "is_desired_behavior": is_desired,
                    "latency_ms": api_result["latency_ms"],
                    "error": None,
                }
            )

            print(f"Outcome: {outcome.value}")

        except Exception as exc:
            print(f"Evaluation failed: {exc}")

            results.append(
                {
                    "item_id": item.id,
                    "benchmark": item.benchmark,
                    "axis": item.axis.value,
                    "model_id": model_id,
                    "raw_response": None,
                    "parsed_outcome": "error",
                    "is_desired_behavior": False,
                    "latency_ms": None,
                    "error": str(exc)
                }
            )

    return results

def main():
    parser = build_parser()
    args= parser.parse_args()

    if args.command == "run":
        axes = [
            axis.strip().lower()
            for axis in args.axes.split(",")
            if axis.strip()
        ]

        # Remove duplicate axes but keep order
        axes = list(dict.fromkeys(axes))

        valid_axes = {"bias", "harm", "factuality", "stereotype"}
        invalid_axes = set(axes) - valid_axes

        if invalid_axes:
            parser.error(
                f"Invalid axes: {', '.join(sorted(invalid_axes))}. "
                f"Choose from: {', '.join(sorted(valid_axes))}"
            )

        if args.limit < 1:
            parser.error("--limit must be at least 1")

        print(f"Preparing {args.model} on {axes}")

        # Load evaluation items
        items = load_selected_items(axes, args.limit)
        print(f"Loaded {len(items)} total evaluation item(s)")

        # Turn items into prompts
        prepared = prepare_prompts(items)
        print(f"Prepared {len(prepared)} total prompt(s)")

        # Call model & score responses
        results = evaluate_prepared_items(prepared, args.model)
        print(f"Completed {len(results)} evaluation(s)")

        report_path = generate_html_report(results, args.out)
        print(f"Saved HTML report to {report_path}")

if __name__ == "__main__":
    main()