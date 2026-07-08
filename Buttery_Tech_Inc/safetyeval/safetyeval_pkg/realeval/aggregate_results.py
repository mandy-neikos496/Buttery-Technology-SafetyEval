import os
import glob
import pandas as pd

log_files = glob.glob("run_*.jsonl")

if not log_files:
    raise FileNotFoundError("No log files found matching 'run_*.jsonl.")

print(f"Found {len(log_files)} run file(s): {', '.join(log_files)}")

def load_logs_as_dataframe(paths: list[str]) -> pd.DataFrame:
    """ 
    Loads and combines every run_*.jsonl file, not just the latest one. If the same (model_id, item_id) pair shows up in more than one file, 
    the newest timestamp wins and older duplicates are dropped.
    """
    frame = [pd.read_json(path, lines=True) for path in paths]
    combined = pd.concat(frame, ignore_index=True)

    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["model_id", "item_id"], keep="last")

    return combined

def compute_axis_reports(df: pd.DataFrame) -> pd.DataFrame:
    scoreable = df[df["parsed_outcome"] != "error"].copy()

    report = (
        scoreable
        .groupby(["model_id", "axis"])
        .agg(
            n_items=("is_desired_behavior", "count"),
            n_desired=("is_desired_behavior", "sum"),
        )
        .reset_index()
    )

    raw_rate = report["n_desired"] / report["n_items"]
    report["desired_rate_%"] = (raw_rate * 100).round(1)

    return report.sort_values(["axis", "model_id"])
        
def count_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Separately surfaces how many calls errored per model."""
    errors = df[df["parsed_outcome"] == "error"]
    if errors.empty:
        return pd.DataFrame(columns=["model_id", "n_errors"])
    return (
        errors
        .groupby("model_id")
        .size()
        .reset_index(name="n_errors")
    )

def build_comparison_grid(report: pd.DataFrame) -> pd.DataFrame:
    """Turns the long-format report into a wide comparison grid: one row per model, one column per axis"""
    grid = report.pivot(index="model_id", columns="axis", values="desired_rate_%")

    grid = grid.fillna("N/A")

    grid = grid.reset_index()
    grid.columns.name = None

    return grid

def analyze_harm_judge_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares the regex-based refusal classifier against LLM judge for harm-axis rows only.
    Returns just the rows where the two disagreed.
    """
    harm_rows = df[df["axis"] == "harm"].copy()

    judged_rows = harm_rows.dropna(subset=["llm_judge_outcome"])

    if judged_rows.empty:
        print("\nNo llm_judge_outcome data found -> judge classifier was not run on this data.")
        return pd.DataFrame()
    
    agree = (harm_rows["parsed_outcome"] == harm_rows["llm_judge_outcome"]).sum()
    total = len(harm_rows)
    print(f"\nRegex vs. LLM judge agreement (harm axis): {agree}/{total} rows agree"
          f"({len(harm_rows) - total} harm row(s) skipped; no judge data)")

    disagreements = harm_rows[harm_rows["parsed_outcome"] != harm_rows["llm_judge_outcome"]]
    return disagreements[["model_id", "item_id", "parsed_outcome", "llm_judge_outcome"]]

def main():
    df = load_logs_as_dataframe(log_files)

    VALID_MODELS = [
        "meta/llama-3.3-70b-instruct",
        "google/gemma-2-2b-it",
        "meta/llama-3.2-1b-instruct",
        "qwen/qwen3.5-122b-a10b",
    ]

    df = df[df["model_id"].isin(VALID_MODELS)].copy()

    print(f"Loaded {len(df)} total log rows (after de-duplicating across files and filtering for valid models)")

    report = compute_axis_reports(df)
    print("\n Aggregate scores (model x axis)")
    print(report.to_string(index=False))

    grid = build_comparison_grid(report)
    print("\n Comparison grid (model x axis)")
    print(grid.to_string(index=False))

    errors = count_errors(df)
    if not errors.empty:
        print("\n Errored calls (excluded from scores above)")
        print(errors.to_string(index=False))

    disagreements = analyze_harm_judge_agreement(df)
    if not disagreements.empty:
        print("\n Disagreements between regex and LLM judge:")
        print(disagreements.to_string(index=False))

    report.to_csv("axis_report_summary.csv", index=False)
    grid.to_csv("comparison_grid.csv", index=False)
    print("\nSaved long-format summary to axis_report_summary.csv")
    print("Saved comparison grid to comparison_grid.csv")

if __name__ == "__main__":
    main()