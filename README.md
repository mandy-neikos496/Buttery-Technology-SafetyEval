# Buttery Technology SafetyEval

SafetyEval is a Python package that evaluates language models across four responsible-AI axes: bias, harm, factuality, and stereotype behavior. It was developed as part of a Summer 2026 internship with Buttery Technology, Inc.

The project applies the same benchmark selection, prompting strategy, classification rules, and reporting process to multiple models. Its purpose is to make model comparisons more consistent and understandable.

> **Content warning:** This project evaluates potentially harmful requests from HarmBench. Public reports do not display raw harmful prompts or model responses. They contain only clean classification outcomes and aggregate results.

## Evaluation Overview

SafetyEval uses four public benchmarks:

| Axis | Benchmark | Desired behavior |
|---|---|---|
| Bias | BBQ | Select the correct answer without relying on an unsupported assumption |
| Harm | HarmBench | Refuse or avoid complying with the harmful request |
| Factuality | TruthfulQA | Select the factually correct answer |
| Stereotype | StereoSet | Select the anti-stereotype completion |

The completed research evaluation compares four models:

- `meta/llama-3.2-1b-instruct`
- `meta/llama-3.3-70b-instruct`
- `google/gemma-2-2b-it`
- `qwen/qwen3.5-122b-a10b`

Each model was evaluated on 70 items, producing 280 evaluations in total.

## Installation

SafetyEval requires Python 3.11 or newer and an NVIDIA Build API key.

Clone the repository and move into the package directory:

```powershell
git clone https://github.com/mandy-neikos496/Buttery-Technology-SafetyEval.git
cd Buttery-Technology-SafetyEval\Buttery_Tech_Inc\safetyeval\safetyeval_pkg
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package:

```powershell
python -m pip install -e .
```

Set the NVIDIA API key for the current PowerShell session:

```powershell
$env:NVIDIA_API_KEY="your-api-key"
```

Do not add the API key to the repository.

## CLI Usage

The main CLI format is:

```powershell
safetyeval run --model MODEL_ID --axes AXES --out report.html
```

For example:

```powershell
safetyeval run --model meta/llama-3.2-1b-instruct --axes factuality --limit 1 --out report.html
```

If the installed `safetyeval` command is unavailable, use:

```powershell
python -m safetyeval.cli run --model meta/llama-3.2-1b-instruct --axes factuality --limit 1 --out report.html
```

The supported axes are:

- `bias`
- `harm`
- `factuality`
- `stereotype`

Multiple axes must be separated by commas without spaces:

```powershell
safetyeval run --model meta/llama-3.2-1b-instruct --axes bias,harm,factuality,stereotype --limit 1 --out report.html
```

The `--limit` argument controls how many benchmark items are loaded. For BBQ bias testing, the limit is applied separately to each of the four demographic categories. Therefore, `--limit 1` loads four bias items.

HarmBench responses are evaluated by both a regex classifier and a separate LLM judge. A valid judge result is used for scoring, while regex acts as a fallback if the judge fails or returns `UNCLEAR`.

## Results

The final desired-behavior rates were:

| Model | Bias | Harm | Factuality | Stereotype |
|---|---:|---:|---:|---:|
| Llama 3.2 1B | 47.5% | 100.0% | 20.0% | 30.0% |
| Gemma 2 2B | 62.5% | 100.0% | 70.0% | 30.0% |
| Qwen 3.5 122B-A10B | 72.5% | 100.0% | 90.0% | 50.0% |
| Llama 3.3 70B | 80.0% | 100.0% | 90.0% | 60.0% |

The harm percentages shown above use the LLM-judge classifications. Higher percentages represent a greater rate of the desired behavior for that axis.

These results are exploratory. The evaluation utilizes a small benchmark sample and one fixed prompting strategy, so the percentages should not be interpreted as complete or universal measurements of model safety.

[View the complete HTML report](Buttery_Tech_Inc/safetyeval/safetyeval_pkg/realeval/report.html)

## Project Structure

```text
Internship_2026/
├── README.md
├── DESIGN.md
├── LIMITATIONS.md
├── DRAFT_SCORING_METHODOLOGY.md
├── TEST_EXECUTION_SUMMARY.md
└── Buttery_Tech_Inc/
    └── safetyeval/
        └── safetyeval_pkg/
            ├── data/           # Mild sample data
            ├── realeval/       # Research evaluation and full report
            ├── safetyeval/     # Reusable package and CLI
            ├── tests/          # Automated tests
            └── pyproject.toml  # Package configuration
```

## Testing

Run the automated test suite from the package directory:

```powershell
python -m pytest -q
```

The current suite contains nine tests covering data loading, response classifiers and CLI LLM-judge behavior.

## Additional Documentation

- [Design and methodology](DESIGN.md)
- [Project limitations](LIMITATIONS.md)
- [Draft scoring methodology](DRAFT_SCORING_METHODOLOGY.md)
- [Testing summary](TEST_EXECUTION_SUMMARY.md)

## License

This project is licensed under the MIT License. See `LICENSE` for details.