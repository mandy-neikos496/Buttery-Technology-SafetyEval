from pathlib import Path

from safetyeval.data_structures import EthicsAxis
from safetyeval.evaluation import run_offline_evaluation
from safetyeval.ingestion import load_items_from_file

SAMPLE_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample-data.json"
)

def test_sample_data_loads():
    items = load_items_from_file(SAMPLE_DATA)

    assert len(items) > 0

    loaded_axes = {item.axis for item in items}

    assert EthicsAxis.BIAS in loaded_axes
    assert EthicsAxis.HARM in loaded_axes
    assert EthicsAxis.FACTUALITY in loaded_axes
    assert EthicsAxis.STEREOTYPE in loaded_axes

def test_offline_evaluation_covers_all_axes():
    items = load_items_from_file(SAMPLE_DATA)
    tally = run_offline_evaluation(items)

    assert tally[EthicsAxis.BIAS]["total"] > 0
    assert tally[EthicsAxis.HARM]["total"] > 0
    assert tally[EthicsAxis.FACTUALITY]["total"] > 0
    assert tally[EthicsAxis.STEREOTYPE]["total"] > 0