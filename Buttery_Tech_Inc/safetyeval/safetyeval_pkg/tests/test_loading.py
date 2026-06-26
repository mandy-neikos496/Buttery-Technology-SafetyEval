from safetyeval.ingestion import load_items_from_file
from safetyeval.evaluation import run_offline_evaluation
from safetyeval.data_structures import EthicsAxis
    
# load items
items = load_items_from_file("../data/sample-data.json")
tally = run_offline_evaluation(items)

# basic checks
assert len(items) > 0
assert tally[EthicsAxis.HARM]["total"] > 0
assert tally[EthicsAxis.BIAS]["total"] > 0
assert tally[EthicsAxis.STEREOTYPE]["total"] > 0
assert tally[EthicsAxis.FACTUALITY]["total"] > 0

# summary
print("\nAxisReport Summary:")
for axis, counts in tally.items():
    total = counts["total"]
    desired = counts["desired"]

    if total > 0:
        pct = (desired / total) * 100
        print(f" {axis.value}: {desired}/{total} ({pct:.1f}%)")