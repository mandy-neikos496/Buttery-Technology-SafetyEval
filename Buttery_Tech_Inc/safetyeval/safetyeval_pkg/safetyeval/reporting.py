from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

def generate_html_report(results, output_path):
    summary = defaultdict(
        lambda: {
            "total": 0,
            "desired": 0,
            "errors": 0,
        }
    )

    for result in results:
        axis = result["axis"]

        if result["parsed_outcome"] == "error":
            summary[axis]["errors"] += 1
            continue

        summary[axis]["total"] += 1

        if result["is_desired_behavior"]:
            summary[axis]["desired"] += 1

    summary_rows = []

    for axis, counts in sorted(summary.items()):
        total = counts["total"]
        desired = counts["desired"]
        errors = counts["errors"]

        rate = (desired / total * 100) if total else 0

        summary_rows.append(
            f"""
            <tr>
            <td>{escape(axis)}</td>
            <td>{total}</td>
            <td>{desired}</td>
            <td>{rate:.1f}%</td>
            <td>{errors}</td>
            </tr>
            """
        )

    result_rows = []

    for result in results:
        desired = (
            "Yes"
            if result["is_desired_behavior"]
            else "No"
        )

        primary_outcome = result["parsed_outcome"]

        judge_outcome = (
            result.get("llm_judge_outcome")
            or "not run"
        )

        scored_outcome = result.get(
            "scored_outcome",
            primary_outcome,
        )

        scoring_source = result.get(
            "scoring_source",
            "primary_classifier",
        )

        result_rows.append(
            f"""
            <tr>
            <td>{escape(str(result["item_id"]))}</td>
            <td>{escape(str(result["benchmark"]))}</td>
            <td>{escape(str(result["axis"]))}</td>
            <td>{escape(str(primary_outcome))}</td>
            <td>{escape(str(judge_outcome))}</td>
            <td>{escape(str(scored_outcome))}</td>
            <td>{escape(str(result["scoring_source"]))}</td>
            <td>{desired}</td>
            </tr>
            """
        )

    document = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SafetyEval Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 40px auto;
                padding: 0 20px;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 32px;
            }}

            th, td {{
                border: 1px solid #cccccc;
                padding: 10px;
                text-align: left;
            }}

            th {{
                background: #eeeeee;
            }}

            .warning {{
                padding: 14px;
                background: #fff3cd;
                border-left: 5px solid #ffcc00;
            }}
        </style>
    </head>

    <body>
        <h1>SafetyEval Report</h1>

        <p>Generated: {escape(datetime.now().isoformat())}</p>

        <div class="warning">
        <strong>Content note:</strong>
        This report contains summary scores and clean outcome labels only.
        Raw HarmBench responses are never displayed.
        </div>

        <h2>Axis summary</h2>

        <table>
            <thead>
            <tr>
            <th>Axis</th>
            <th>Scoreable items</th>
            <th>Desired outcomes</th>
            <th>Desired-behavior rates</th>
            <th>Errors</th>
            </tr>
            </thead>
            <tbody>
                {''.join(summary_rows)}
            </tbody>
        </table>

        <h2>Individual results</h2>

        <table>
            <thead>
            <tr>
            <th>Items</th>
            <th>Benchmark</th>
            <th>Axis</th>
            <th>Primary outcome</th>
            <th>LLM-judge outcome</th>
            <th>Scored outcome</th>
            <th>Scoring source</th>
            <th>Desired behavior?</th>
            </tr>
            </thead>
            <tbody>
                {''.join(result_rows)}
            </tbody>
        </table>
    </body>
    </html>
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")

    return path.resolve()