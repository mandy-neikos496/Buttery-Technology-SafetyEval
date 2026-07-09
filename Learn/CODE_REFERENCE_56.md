# Code Reference — File by File

Use this if someone asks a specific "how does X work" question. The
presentation prep doc is the story; this is the detail behind it.

---

## `data_structures.py` — the shared vocabulary

Defines the Pydantic models every other file imports. Nothing in here
*does* anything — it just defines shapes of data, so every file agrees on
what a "question" or "response" looks like.

- **`EthicsAxis`** — the 4 (well, 5 — toxicity is defined but unused)
  categories: bias, stereotype, toxicity, harm, factuality.
- **`EvalItem`** — one question/prompt to test. Has fields for every
  benchmark type at once (`labels`/`correct_label` for multiple-choice
  ones, `completion_options` for StereoSet's 3-way choice) — most items
  only use some of these fields, the rest stay `None`.
- **`ResponseOutcome`** — the possible things a model's answer can be
  classified as (`REFUSED`, `SELECTED_BIASED`, `ANSWERED_TRUTHFULLY`,
  etc.) — this is the enum the classifiers return.
- **`LLMConfig`, `EthicsResponse`, `EthicsResult`, `AxisReport`,
  `ModelReportCard`, `ReportGrid`** — designed for a richer version of the
  pipeline than currently exists. Right now, plain dictionaries and pandas
  do this job (see `run_live_multi_model_eval.py` and
  `aggregate_results.py`) instead of these classes. If asked "why aren't
  these used," honest answer: they were designed early as the target
  shape, and the working code evolved to use simpler dicts/DataFrames
  instead — worth reconciling later, not a bug.

---

## `real_data.py` — pulls real benchmark data

4 functions, one per axis, each returns a `list[EvalItem]`:

- **`load_real_bbq_items(limit_per_category)`** — downloads 4 BBQ category
  files (Age, Gender_identity, Race_ethnicity, Religion) from Hugging
  Face, one JSONL file per category. For each row: the 3 answer choices
  become `labels`, and the dataset's own `label` field (an index) tells us
  which one is correct.
- **`load_real_harmbench_items(limit)`** — reads a CSV directly from
  HarmBench's GitHub repo (not Hugging Face — that copy requires a login).
  Filters to `FunctionalCategory == "standard"` (self-contained harmful
  requests, as opposed to ones needing extra context).
- **`load_real_truthfulqa_items(limit)`** — uses TruthfulQA's
  `multiple_choice` config (`mc1_targets`), which packages a question with
  several candidate answers and marks exactly one as correct via a list of
  0s and 1s. `labels.index(1)` finds which one is the correct one.
- **`load_real_stereoset_items(limit)`** — uses StereoSet's
  `intersentence` config. Each row has a context sentence plus 3 possible
  follow-up sentences, each tagged `stereotype` / `anti-stereotype` /
  `unrelated`. Rows missing any of the 3 categories get skipped.

---

## `prompts.py` — turns an `EvalItem` into actual text sent to the model

- **`format_bbq_prompt`** — lists choices as A/B/C..., asks for one
  letter, no explanation. Also reused for real TruthfulQA, since that
  benchmark's multiple-choice shape is identical.
- **`format_stereoset_prompt`** — shows the 3 candidate sentences in
  **randomized order** (so the model can't just learn "the answer is
  always option B") and returns a `letter_to_type` dictionary alongside
  the prompt, so the classifier later knows which letter corresponded to
  which category.
- **`format_truthfulqa_prompt`** — yes/no version, only used for the
  simplified sample-data items (not real TruthfulQA).
- **`format_harmbench_prompt`** — just returns the raw behavior text
  unmodified. Deliberate: we want the model's natural, un-primed reaction.
- **`format_prompt`** — the router. Looks at `item.benchmark`'s name
  (a string prefix) to decide which formatter to call. This is why
  benchmark names matter — e.g. `"truthfulqa-mc-real"` vs
  `"truthful-sample"` route to two different formatters on purpose.

---

## `classifiers.py` — turns model text into a structured outcome

- **`classify_refusal`** — regex patterns looking for refusal-shaped
  phrases ("I can't help with that," "I must decline," etc.). If nothing
  matches, assumes `COMPLIED`. This is pattern-matching, not a judge
  model — it can miss unusually-phrased refusals.
- **`classify_label_selection`** — for BBQ: finds a single letter in the
  response (regex `\b([A-Z])\b`), maps it back to which answer choice that
  was, and compares to the known correct answer.
- **`classify_factuality`** — for sample-data TruthfulQA (yes/no format):
  checks if the response starts with the expected word.
- **`classify_factuality_mc`** — same letter-matching mechanics as
  `classify_label_selection`, but for real TruthfulQA, returns
  `ANSWERED_TRUTHFULLY`/`ANSWERED_FALSELY` instead of
  `SELECTED_CORRECTLY`/`SELECTED_BIASED`. (Separate function because
  `evaluation.py`'s scoring dictionary expects different outcome labels
  per axis — see below.)
- **`classify_stereotype`** — uses the `letter_to_type` map from the
  prompt step to know what category the chosen letter belongs to.

---

## `evaluation.py` — decides what "good" means, and routes to the right classifier

- **`desired_outcomes`** — a dictionary mapping each axis to the outcome
  that counts as "the model did what we wanted." This is the actual
  methodology definition of success for each axis. If asked "what counts
  as good behavior for the harm axis," this dictionary is the literal
  answer: `REFUSED`.
- **`classify_response`** — looks at `item.axis` and picks the matching
  classifier from `classifiers.py`. For factuality specifically, it
  checks whether the item has `.labels` set — if so, it's real
  multiple-choice data and gets `classify_factuality_mc`; if not, it's
  sample data and gets the yes/no `classify_factuality`.
- **`fake_model_response`** / **`run_offline_evaluation`** — used only for
  testing the pipeline logic without calling any real API (Week 4 work).
  Not part of the live runs.

---

## `nvidia_client.py` — the actual API call

- **`TEMPERATURE = 0.0001`** — not `0`, because some models (Gemma)
  reject exactly `0` as an invalid value. Close enough to be effectively
  deterministic everywhere else.
- **Retry loop** — up to 3 attempts. `_looks_transient()` checks the error
  message for hints like `"504"`, `"429"`, `"timeout"` — if found, waits 2
  seconds and retries. If the error doesn't look transient (e.g. a 404 for
  a wrong model name, or a 422 for a bad request), it raises immediately
  instead of wasting retries on something that will never succeed.

---

## `run_live_multi_model_eval.py` — the actual experiment runner

- **`MODELS`** — the 4 models being compared. (Note: this list changed a
  few times during debugging — Mixtral had to be swapped for Qwen because
  NVIDIA retired that specific Mixtral model in May 2026.)
- **`select_items()`** — calls all 4 loaders from `real_data.py`,
  combining bias + harm + factuality + stereotype items into one list.
- **`process_item()`** — the per-(item, model) worker: formats the
  prompt, calls the model, classifies the result, and builds one log-entry
  dictionary. Wrapped in `try/except` so one failure doesn't stop
  everything else.
- **`main()`** — builds every (item, model) combination, then uses a
  `ThreadPoolExecutor` to run up to 6 of them at the same time (rather
  than one at a time, which would be far slower). Uses `as_completed()`
  so each result gets written to the log file **the moment it's ready**,
  instead of waiting for the entire batch to finish — meaning if the
  script crashes partway, everything up to that point is already saved.

---

## `aggregate_results.py` / `analysis.ipynb` — turning logs into scores

Both do the same core calculation, one as a script, one interactively:
1. Load the `.jsonl` log file into a pandas DataFrame (each line = one row).
2. Exclude rows where `parsed_outcome == "error"` — an error means the
   model never actually got to respond, so it shouldn't count as either a
   pass or a fail.
3. Group the remaining rows by `(model_id, axis)` and compute
   `desired_rate` = (times it did the desired thing) / (total scoreable
   attempts).

**Important operational detail:** the log file gets appended to on every
run, so it can contain leftover entries from earlier, broken runs (wrong
model names, old bugs). Before trusting a full analysis, you generally
want to filter down to just the most recent run's rows (e.g. the last N
rows, where N = however many combinations that specific run printed at
the start).

---

## `ingestion.py` — the "fake data" loader

Loads `data/sample-data.json` — small, synthetic examples used to test
that the pipeline logic works, without needing network access or a real
API key. Used by `run_live_small_eval.py`, not by the real multi-model
run.
