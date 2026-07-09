# Your Project, in Plain English (Presentation Prep)

## The one-sentence pitch
"I built a pipeline that tests real LLMs against real academic safety/bias
benchmarks — bias, harm, factuality, and stereotypes — the same way a
research paper would, so we can actually compare how safe different models
are instead of just trusting each provider's own marketing."

If you say nothing else, say that. Everything below just backs it up.

---

## The pipeline as a story (6 steps, 6 files)

Think of it as an assembly line. Each file below is one station. Data
flows through all of them in order, once per (item, model) pair.

### 1. Get the questions — `real_data.py`
Each of the 4 axes has a real, published academic benchmark behind it:
- **Bias** → BBQ (from Hugging Face)
- **Harm** → HarmBench (from the authors' GitHub — real adversarial prompts)
- **Factuality** → TruthfulQA (Hugging Face, multiple-choice version)
- **Stereotype** → StereoSet (Hugging Face)

Each loader function downloads real examples and turns them into your
`EvalItem` shape (defined in `data_structures.py`) — a standard format so
the rest of the pipeline doesn't need to know or care which benchmark an
item came from.

**If asked "why real data and not made-up examples":** SPEC.md's own
guidance was to prototype with sample data first, then move to real
datasets once the pipeline logic was proven — sample data was for testing
the pipeline, real data is for actual results.

### 2. Turn a question into a prompt — `prompts.py`
Different benchmarks need different question formats:
- BBQ / TruthfulQA → "here's a question and lettered choices, answer with
  one letter"
- StereoSet → "here's a passage and 3 possible endings, pick the best one"
- HarmBench → sent as-is (we want to see the model's natural reaction to
  a raw request)

**Key detail worth mentioning:** TruthfulQA's real multiple-choice format
turned out to be *exactly* the same shape as BBQ's (question + lettered
choices + one correct answer), so that formatter gets reused instead of
duplicated. That's an actual design decision, not an accident — same code,
two different benchmarks.

### 3. Ask the model — `nvidia_client.py`
Sends the prompt to NVIDIA's API and gets a response back. Two
non-obvious things live here that are worth being able to explain:
- **Temperature is `0.0001`, not `0`.** Some models (Gemma, specifically)
  reject `temperature=0` outright as invalid. `0.0001` is close enough to
  be effectively deterministic for every model, but doesn't get rejected.
- **Automatic retries.** If a call fails with something that looks
  temporary (a timeout, a "server busy" error), it waits a couple seconds
  and tries again — up to 3 times — before giving up. But if the error is
  permanent (wrong model name, invalid request), it fails immediately
  instead of wasting time retrying something that can never succeed.

### 4. Score the response — `classifiers.py` + `evaluation.py`
Turns raw model text into a structured outcome:
- Did it pick the correct/non-biased answer?
- Did it refuse the harmful request, or comply?
- Did it answer the factual question truthfully?

`evaluation.py` also defines what "desired behavior" means per axis (e.g.
for harm, desired = refused; for bias, desired = picked the correct
answer) — this is the actual scoring logic, and it's worth being able to
say out loud what counts as "good" for each axis, since that's a real
methodology decision, not an obvious one.

### 5. Run it all, and log everything — `run_live_multi_model_eval.py`
Loops over every model × every item, calls the model, classifies the
result, and writes one line to a `.jsonl` log file per attempt — as soon
as each one finishes (not batched at the end), using multiple requests in
parallel to keep it from taking forever.

**Safety detail worth mentioning:** raw HarmBench responses are never
written to the log at all — they're replaced with `"[hidden for safety]"`
before saving. This was a requirement from the start (SPEC.md), not
something bolted on later: only whether the model complied or refused
gets recorded, never what a "compliant" answer actually said.

### 6. Turn logs into scores — `aggregate_results.py` / `analysis.ipynb`
Loads the log file into a pandas table and groups it by model and axis to
compute a `desired_rate` — e.g. "Model X did the desired thing 80% of the
time on the bias axis." This is the actual comparison your project set
out to produce.

**Detail worth mentioning:** rows where the API call errored are excluded
from scoring, not counted as failures — an error means the model never
actually got to answer, so it shouldn't count for or against it.

---

## Likely questions and how to answer them

**"Why these 4 axes / these 4 benchmarks?"**
They're established, peer-reviewed benchmarks already used in the
literature (cite SPEC.md's paper links if pushed) — the goal was
standardizing evaluation, not inventing new tests.

**"Why these 4 models?"**
Deliberately different sizes and origins (a 70B and a 1B from the same
family, a Google model, a non-Western-trained model) — comparing similar
models tells you less than comparing different ones.

**"What does a 'desired rate' of 70% actually mean?"**
Out of the items that axis was tested on, the model did the
hoped-for thing 70% of the time. It does NOT mean the model is "70% safe"
overall — it's specific to that one axis, that one benchmark, that one
prompting style.

**"What are the limitations?"** (Good to have 2-3 ready)
- Small sample size per axis (be honest about your actual `ITEMS_PER_AXIS`
  number) — a few different answers would meaningfully swing the percentage.
- One fixed prompting style — a differently-worded prompt could change
  results.
- Refusal detection is regex-pattern-based, not a judge model — it might
  miss unusual refusal phrasing.

If you don't know an answer live, "that's actually one of the things I
want to dig into more before the final report" is a completely legitimate
answer — it shows you understand what's unresolved, which is exactly what
SPEC.md's `LIMITATIONS.md` section is asking you to be good at.
