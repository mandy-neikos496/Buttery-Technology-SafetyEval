# MASTER STUDY GUIDE — Read This One, Top to Bottom

Organized by time budget. Do Tier 1 no matter what — it alone is enough to
survive most of the presentation. Add tiers as time allows.

---

# TIER 1 — If you only have 10 minutes

## The pitch (memorize this, word for word if you need to)
"I built a pipeline that tests real LLMs against real academic safety/bias
benchmarks — bias, harm, factuality, and stereotypes — so we can compare
how safe different models actually are, instead of trusting each
provider's own marketing."

## The pipeline = 6 steps. Say them in order, and you've explained the whole project.

1. **Get real questions** (`real_data.py`) — downloads real benchmark
   items from Hugging Face / GitHub: BBQ (bias), HarmBench (harm),
   TruthfulQA (factuality), StereoSet (stereotype).
2. **Format a prompt** (`prompts.py`) — turns a question into the exact
   text sent to the model (e.g. "pick a letter A-D").
3. **Call the model** (`nvidia_client.py`) — sends it to NVIDIA's API,
   gets a response back.
4. **Score the response** (`classifiers.py` + `evaluation.py`) — decided
   if the model did the "desired" thing (refused, picked the right
   answer, etc.).
5. **Run it all + log it** (`run_live_multi_model_eval.py`) — loops over
   every model × every question, saves each result to a file as it
   finishes.
6. **Turn logs into scores** (`aggregate_results.py` / `analysis.ipynb`) —
   pandas computes "% of the time each model did the desired thing," per
   model, per axis.

## The 4 models you're comparing, and why
- `meta/llama-3.3-70b-instruct` — large, open-source
- `meta/llama-3.2-1b-instruct` — small, same family (does size affect safety?)
- `google/gemma-2-2b-it` — different company, safety-tuned
- `mistralai/mistral-large-3-675b-instruct-2512` — different architecture/company

**If asked "why these 4":** different sizes and companies on purpose —
comparing similar models tells you less than comparing different ones.

## If you say nothing else about limitations, say this:
"This is a small sample size per axis, and one fixed prompting style — a
different prompt could change results. That's exactly what my
`LIMITATIONS.md` is for: being upfront about what this does and doesn't prove."

---

# TIER 2 — If you have 30 minutes: the "why," not just the "what"

These are the questions that separate "I copied working code" from "I
understand this" — worth having 3-4 of these ready.

**Q: What does "desired behavior" mean, exactly?**
A: It's axis-specific, defined in `evaluation.py`'s `desired_outcomes`
dictionary: for harm, desired = the model refused. For bias/stereotype,
desired = it picked the non-biased answer. For factuality, desired = it
answered correctly. Same model, different axis, different definition of
"good" — that's a real methodology decision, not an obvious default.

**Q: Why do you hide raw harm-axis responses?**
A: Per the project's safety guidelines: we log *whether* a model complied
or refused with a harmful request, never *what* a compliant answer
actually said. In `run_live_multi_model_eval.py`:
```python
safe_raw_response = "[hidden for safety]" if item.axis == EthicsAxis.HARM else raw_text
```
The real text exists briefly in memory (needed to classify refused vs.
complied) but is swapped out before anything touches disk.

**Q: Why real datasets instead of made-up test questions?**
A: The project's own early guidance was: prototype the pipeline logic with
small sample data first, then swap to real, published benchmarks once the
logic is proven. Sample data tests the code; real data produces the
actual results.

**Q: How does scoring actually work — like, mechanically?**
A: Mostly regex-based pattern matching, not another AI model judging the
answer. For multiple-choice axes (bias, factuality-real, stereotype): find
a single letter in the response, compare it to the known correct answer.
For harm: search the text for refusal-shaped phrases ("I can't help with
that," etc.) — if none match, assume it complied.

**Q: What's a real limitation of that scoring approach?**
A: The refusal detector is pattern-matching, not a judge model — it could
miss an unusually-phrased refusal and wrongly mark it as "complied." Also,
small sample sizes (per SPEC guidance, likely single digits per axis)
mean one different answer can swing a percentage a lot.

---

# TIER 3 — If you have more time / want to sound extra sharp: the debugging story

This is genuinely good material — it shows real engineering, not just a
finished result. Pick 2-3 you actually remember clearly enough to explain
without notes.

| What happened | Why | What it shows |
|---|---|---|
| Every call to one model returned `Error code: 410 ... end of life` | The model had been genuinely retired by the provider | Model catalogs change over time — don't assume a model ID is permanent |
| Every call to another model returned `422 ... temperature must be greater than 0` | `temperature=0` (which most models accept fine) was rejected by this specific model's validation | Not every provider treats "the same" parameter identically |
| Every call to a model returned `404` | A typo in the model ID string (dash instead of dot) | A 404 always means "this identifier doesn't exist" — check spelling before assuming a bigger problem |
| One call crashed with `'NoneType' object has no attribute 'strip'` | A "reasoning" model spent its whole token budget thinking internally, leaving no visible answer | APIs don't always return the shape you expect — defensive code matters |
| One model timed out on every attempt, across 3 separate runs | The provider's endpoint for that specific model seemed to be down/overloaded | Not everything is a bug in *your* code — sometimes it's a real external outage, and that's a legitimate thing to document rather than keep chasing |
| Runs were very slow with high concurrency | Too many simultaneous API calls caused timeout errors | Deliberately traded speed for reliability — lowered concurrent requests to reduce errors, at the cost of longer total runtime |

**The meta-point, if asked "what was hardest":** most of today's problems
weren't logic bugs in the code I wrote — they were real-world integration
issues (wrong model names, provider-specific quirks, temporary outages).
That's a normal, expected part of working with live external APIs, not a
sign something was designed badly.

---

# TIER 4 — Quick-lookup reference (only open this if someone asks a specific "how does X work")

- **`data_structures.py`** — just defines data shapes (Pydantic models),
  doesn't *do* anything. `EvalItem` = one question, `ResponseOutcome` =
  possible ways a response gets classified.
- **`real_data.py`** — 4 functions, one per axis, each downloads/loads
  real benchmark items and returns them in the shared `EvalItem` format.
- **`prompts.py`** — `format_prompt()` is the router; looks at the
  benchmark's name to decide which exact wording to send the model.
- **`classifiers.py`** — the actual pattern-matching logic (regex for
  letters, regex for refusal phrases) that turns text into a `ResponseOutcome`.
- **`evaluation.py`** — `desired_outcomes` dict = the methodology
  definition of "good" per axis. `classify_response()` routes to the
  right classifier.
- **`nvidia_client.py`** — the actual API call, with retry logic for
  temporary errors and a guard against empty responses.
- **`run_live_multi_model_eval.py`** — the runner: builds every
  (model × question) combination, runs them with limited parallelism,
  saves each result the moment it's ready.
- **`aggregate_results.py` / `analysis.ipynb`** — pandas: load the log
  file, exclude errored rows, group by model+axis, compute % desired.

---

# Last resort, if you blank on something live:

"That's a good question — I want to double check the exact logic before I
answer definitively rather than guess" is a completely fine thing to say.
It's honest, and it's a better answer than confidently saying something
wrong. You are allowed to not have every line memorized; you built a real,
working pipeline today that handles real API failures gracefully — that's
the actual accomplishment.
