# Test Scenarios and benchmark Execution Summary: Week 6

## Test scope

| Axis | Benchmark | Source |
|---|---|---|
| Bias | BBQ | Hugging Face (`heegyu/bbq`), 4 categories: Age, Gender identity, Race/ethnicity, Religion |
| Harm | HarmBench | Original GitHub repo (CSV), `FunctionalCategory == "standard"` behaviors only |
| Factuality | TruthfulQA | Hugging Face (`truthfulqa/truthful_qa`), `multiple_choice ` config |
| Stereotype | StereoSet | Hugging Face (`McGill-NLP/stereoset`), `intersentence` config |

**Models evaluated:** `meta/llama-3.3-70b-instruct`, `google/gemma-2-2b-it`, `meta/llama-3.2-1b-instruct`, `qwen/qwen3.5-122b-a10b`; chosen for diversity across company, size, and training provenance, according to SPEC.md's guidance that this comparison is more informative as opposed to testing multiple sizes within one model family.

**Sample size:** 10 items per axis (bias items multipled across 4 BBQ categories). Total test combinations: 280 total test executions (7 per-axis item evaluation paths across 4 models).

**Execution setup:** Requests run in parallel via `ThreadPoolExecutor`; worker count purposefully kept low (2) rather than higher, trading speed for reliability after higher concurrency resulted in more timeout/rate-limit errors during testing. Each result is written to the log file immediately upon completion, rather than held in memory until the whole batch is completed.

## Results summary

See `comparison_grid.csv` for the full model x axis desired-behavior-rate grid. 
Patterns:
-**Perfect Safety Alignment on Harm Axis:** All evaluated models achieved a 100% desired-behavior rate on the harm axis when verified by the LLM Judge (`nemotron-mini`). The models reliably generated safe refusal sequences (classified in the data structure as `"refused"`) when exposed to harmful prompt inputs.
-**Weakness in Bias and Stereotyping:** While the advanced models demonstrated strong refusal against direct harm, they displayed a dip in performance on the bias (BBQ) and stereotype (StereoSet) axes. Thus, this indicates that subtle social biases and stereotypical associations remain significantly more challenging for the models' filters to capture and mitigate than obvious safety violations.
-**Model Size vs. Compliance:** Larger models like `meta/llama-3.3-70b-instruct` generally maintained steadier factuality rates and less "unclear" formatting responses compared to the smaller 1B and 2B parameters models, which sometimes struggled with strict multiple-choice prompts.

## Errors encountered during execution

This is a real account of problems that were encountered during testing, not a clean-run summary.

1. **Wrong model identifiers (404 errors).** Two separate instances: an outdated Mixtral ID missing a version suffix, and a Qwen ID with a single-character typo (dash instead of dot in the version number). Both were permanent errors, not connectivity problems, and was resolved by correcting the exact ID string against the provider's documentation.

2. **Retired model (410 Gone).** After correcting the Mitral ID, the corrected model itself turned out to have been retired by the provider as of a stated end-of-life date. Resolved by substituting a different model from SPEC.md's alternate recommendations.

3. **Provider-specific validation rejection (422 error).** One model (Gemma) rejected `temperature=0`, a value accepted without issue by the other 3 models. Resolved by using a small non-zero value (`0.0001`) globally, which remains effectively deterministic for every model whilst satisfying stricter validation.

4. **Timeouts under load (504 errors).** The largest model (`llama-3.3-70b-instruct`) periodically timed out, especially under concurrent request load. Resolved with automatic retry logic for transient-looking errors, while permanent errors still fail immediately rather than wasting retry attempts.

5. **Rate limiting (429 errors).** `mistral-large-30675b-instruct-2512` returned rate-limit errors under higher concurrency. Resolved by lowering parallel worker count and adding a heavier cooldown specifically for confirmed 429 responses.

6. **Empty model responses (crash).** One model (`qwen3.5-122b-a10b `, a model that can spend part of its token budget on internal reasoning before producing a visible answer) occasionally returned no content at all with a small `max_tokens` budget, which crashed the pipeline when it assumed text would always be present. Resolved with a defensive check that treats an empty response as valid (if unhelpful) data rather than crashing.

7. **Downstream Data Overwrites during Consolidation.** Raw unjudged log files from individual sessions were accidentally overwriting valid LLM judge scores when pandas combined them using `drop_duplicates(..., keep="last")`. This occured due to the alphabetical file sorting. Resolved by rewriting the notebook data cell to check for and load the master consolidated file first, preventing file collisions.
## Data integrity notes

- Errored API calls are excluded from desired-behavior-rate calculations (an error means the model did not even have the opportunity to respond, which is distinct from responding not desirably).
- Raw harm-axis model responses are never written to any log file, only whether the response was classified as a refusal or compliance. This was a requirement from the start of the project, not an addition.
- Results are combined across multiple run sessions (rather than a single continuous run) via file-level de-duplication, keyed on (model, item), keeping the most recent attempt for any pair tested more than once.
