from safetyeval.ingestion import load_items_from_file
from safetyeval.evaluation import run_offline_evaluation
    
# load items
items = load_items_from_file("../data/sample-data.json")
tally = run_offline_evaluation(items)

# summary
print("\nAxisReport Summary:")
for axis, counts in tally.items():
    total = counts["total"]
    desired = counts["desired"]

    if total > 0:
        pct = (desired / total) * 100
        print(f" {axis.value}: {desired}/{total} ({pct:.1f}%)")