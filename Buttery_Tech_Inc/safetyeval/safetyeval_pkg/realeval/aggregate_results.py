import os
import glob
import pandas as pd

log_files = glob.glob("run_*.jsonl")

if not log_files:
    raise FileNotFoundError("No log files found matching 'run_*.jsonl.")

LOG_FILE = max(log_files, key=os.path.getctime)
print(f"Automatically analyzing the latest run file: {LOG_FILE}")

def load_logs_as_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df


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
    report["desired_rate"] = (report["n_desired"] / report["n_items"]).round(3)

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

def main():
    df = load_logs_as_dataframe(LOG_FILE)
    print(f"Loaded {len(df)} total log rows from {LOG_FILE}")

    report = compute_axis_reports(df)
    print("\n Aggregate scores (model x axis)")
    print(report.to_string(index=False))

    errors = count_errors(df)
    if not errors.empty:
        print("\n Errored calls (excluded from scores above)")
        print(errors.to_string(index=False))

    report.to_csv("axis_report_summary.csv", index=False)
    print("\nSaved summary to axis_report_smmary.csv")

if __name__ == "__main__":
    main()