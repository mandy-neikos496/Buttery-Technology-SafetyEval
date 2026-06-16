import json
from safetyeval.data_structures import EvalItem

def load_items_from_file(path: str) -> list[EvalItem]:
    with open(path) as f:
        data = json.load(f)

    items = []
    for benchmark in data["benchmarks"]:
        for item_dict in benchmark["items"]:
            items.append(EvalItem(**item_dict))
    return items