# Development Progress Report: Week 6

## Summary:

This week’s goal was executing real benchmark runs across multiple models and building the infrastructure to score plus compare them. Originating from a working single-model off of sample data, this week extended the safetyeval project to: real benchmark data across all 4 axes, 4 diverse models, pandas-based aggregate scoring, and an exploratory analysis notebook, along with an LLM-as-judge comparison for the harm axis.

## What was built:
**Real data loaders (`real_data.py`):** 4 real loaders (HarmBench for the harm-axis, TruthfulQA for the factuality-axis, StereoSet for the stereotype-axis, and BBQ for the bias-axis) were consolidated into one file. TruthfulQA specifically required a design decision in which its real multiple choice format needed different prompt/scoring logic than the simplified yes/no format utilized within earlier sample data, so a parallel `classify_factuality_mc()` classifier was added, and `evaluation.py` routes to it automatically based on whether an item has multiple choice labels attached.

**Multi-model runner (`run_live_multi_model_eval.py`):** built to run every combination (model x benchmark item), using parallel requests (`ThreadPoolExecutor`) and writing each result to a log file the moment it is done (`as_completed()`), as opposed to waiting until the end. Thus, a crash or interruption partway through never loses work that is already completed.

**Strong API client (`nvidia_client.py`):** went through several iterations in response to real issues (see Test Execution Summary for details): a small non-zero temperature to satisfy stricter model validation, a guard against models returning empty responses, and retry logic that captures exponential backoff for transient errors (timeouts, rate limits) whilst still failing fast on permanent errors rather than wasting retries on something that will not be able to succeed.

**Aggregation (`aggregate_results.py`, `analysis.ipynb`):** combines all run log files, de-duplicates by (model, item) keeping the newest attempt, computes a per-(model x axis) desired-behavior rate, and produces both a long-format CSV along with a pivoted comparison grid (models as rows, axes as columns) matching realeval’s goal.

**Repair tooling (`patch_eval.py`):** a separate script to retry only the rows that errored out in a previous run, rather than needing to re-run a whole batch to fix a handful of failures.

**LLM-as-judge, harm axis (stretch):** added a second classifier for the harm axis that asks a different model (`nemotron-mini`, purposefully not one of the 4 evaluated models, to avoid a model judging its own output) whether a response refused or complied, run alongside the existing regex-based classifier rather than replacing it. Both results are logged per row for direct comparison.

## Current Status

- 4 real benchmarks integrated across all 4 MVP axes (bias, harm, factuality, stereotype).
- 4 diverse models evaluated: `llama-3.3-70b-instruct` (large, US), `gemma-2-2b-it` (small, US), `mistral-large-3-675b-instruct-2512` (large, EU), `qwen3.5-122b-a10b` (large, non-Western training distribution); chose purposefully for source/size diversity according to SPEC.md’s guidance that this matters more than a same-family size-only comparison.
- Aggregate scoring and comparison-grid output working end to end.
- LLM-as-judge comparison implemented for the harm axis; initial results display  !!!!!!!!!!!!!!!!!!!!!!!!!!!!! write after get results !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## Known gap

`meta/llama-3.2-1b-instruct` was originally intended as a same-family size comparison against `llama-3.3-70b-instruct` (to test if scale affects safety). Its endpoint returned persistent timeout errors across multiple separate attempts, spanning several hours, including an isolated single-call test run to check for recovery. Yet, never once did it succeed. This has been documented as an infrastructure limitation rather than continuing to re-attempt it, and is tracked as a candidate for retry at a later date rather than a blocker.

## Next steps

- Increase sample size per axis (currently a small pilot size) for more statistically meaningful results.
- Investigate the regex vs. judge disagreement pattern found on the harm axis and document findings.
- Continue toward Week 7 deliverables: charts, HTML/Markdown report generation, design writeup.

