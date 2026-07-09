# Presentation Log: Every Error Hit, and How It Was Fixed

Chronological order — this is roughly the order things actually happened,
which also works well as a presentation narrative: "here's the pipeline,
here's what went wrong building it, here's how I fixed each thing."

For each one: **what happened** (the actual error), **why** (root cause),
**the fix**, and **what to show on screen** if you want to point at real
code.

---

### 1. Mixtral: wrong model ID (404)

**What happened:** Every single call to `mistralai/mixtral-8x22b-instruct`
returned `404 page not found`.

**Why:** A 404 means the exact model identifier doesn't exist at that
endpoint — not a connection problem. The real ID needed a version suffix.

**Fix:** Changed the model string to `mistralai/mixtral-8x22b-instruct-v0.1`.

**Show:** The `MODELS` list in `run_live_multi_model_eval.py`.

---

### 2. Mixtral (v0.1 this time): retired model (410 Gone)

**What happened:** After fixing #1, every call to the corrected Mixtral ID
returned `Error code: 410 ... "has reached its end of life on
2026-05-21T00:00:00Z and is no longer available."`

**Why:** NVIDIA had genuinely retired that specific model. Not a bug —
model catalogs change over time.

**Fix:** Swapped to `qwen/qwen3.5-122b-a10b` (an alternate model
`SPEC.md` itself recommends, chosen partly to preserve the project's goal
of testing a "non-Western training distribution" model).

**Show:** `MODELS` list again — good moment to mention "model catalogs
aren't permanent infrastructure."

---

### 3. Gemma: rejected temperature=0 (422)

**What happened:** Every call to `google/gemma-2-2b-it` returned
`422 ... 'temperature\n  Input should be greater than 0'`.

**Why:** The pipeline used `temperature=0` for reproducible, deterministic
outputs — works fine on most models, but Gemma's API validation rejects
exactly `0` as invalid.

**Fix:** Changed `TEMPERATURE` from `0` to `0.0001` — close enough to be
effectively deterministic everywhere, but passes every model's validation.

**Show:** `TEMPERATURE = 0.0001` at the top of `nvidia_client.py`.

---

### 4. Llama-3.3-70b: timeouts under load (504)

**What happened:** Several calls to the 70B model returned
`Error code: 504` (Gateway Timeout), especially once more parallel
requests were running at once.

**Why:** A 504 is typically transient — the server was too slow/busy to
respond in time, not a permanent problem. Large models are simply slower
per call, and get worse under concurrent load.

**Fix:** Added automatic retry logic: on a transient-looking error, wait
and try again (up to several attempts) before giving up.

**Show:** The retry loop in `nvidia_client.py` (`for attempt in
range(1, MAX_RETRIES + 1): ... except Exception as exc: ...`).

---

### 5. Qwen: another wrong-ID issue, subtler this time (404)

**What happened:** Every call to Qwen returned `404`, even after "fixing"
the ID once already (a typo'd version had a dash in the wrong place:
`qwen.3-5-122b-a10b` instead of `qwen3.5-122b-a10b`).

**Why:** Same root cause as #1 — a 404 always means the exact string is
wrong. This one was sneaky because it was *almost* right, one character
out of place.

**Fix:** Corrected the exact string, character by character, against
NVIDIA's own documentation.

**Show:** Good opportunity to say "I learned to double check exact model
ID spelling character-by-character before assuming a bigger problem."

---

### 6. Qwen: crash on empty response

**What happened:** One specific call to Qwen crashed with
`'NoneType' object has no attribute 'strip'`, instead of returning a
normal error.

**Why:** Qwen3.5 is a "reasoning" model — it can spend part of its token
budget on invisible internal reasoning before writing a visible answer.
With a small `max_tokens=16` (deliberately kept cheap), it's possible for
a model to use the *entire* budget thinking, leaving nothing for an
actual answer -- so the response comes back as `None` instead of text.
Calling `.strip()` directly on `None` crashes.

**Fix:** Added a guard: `raw_text = content.strip() if content else ""`
-- treats an empty response as valid (if unhelpful) data, not a crash.

**Show:** That one line in `nvidia_client.py`. Good example of "APIs
don't always return the exact shape you expect, even when nothing is
technically wrong -- defensive code matters."

---

### 7. "Saved 0 log entries" despite the run clearly working

**What happened:** After switching the script to save each result
immediately instead of batching everything at the end, the final summary
line incorrectly printed `Saved 0 log entries`.

**Why:** Leftover code from the older version -- an unused `logs = []`
list was still being measured with `len(logs)`, instead of the new
running `completed` counter that actually tracked progress.

**Fix:** Changed the print statement to use `completed` instead.

**Show:** Good talking point on its own: "refactors can leave orphaned
leftover code that looks harmless but reports wrong information -- worth
re-reading a whole function after a structural change, not just the part
you meant to edit."

---

### 8. Manual merge typos (syntax error + old bug reappearing)

**What happened:** After manually combining several fixes by hand, the
file wouldn't even run: `"(" was not closed"` on the line with
`temperature=TEMPERATURE` / `max_tokens=16`.

**Why:** Three separate small mistakes: a missing comma between two
function arguments (which confuses Python badly enough that it blames the
wrong line), `MAX_RETIRES` typo'd instead of `MAX_RETRIES`, and the
`.strip()`-on-`None` crash from #6 had silently crept back in because the
`return` statement was still using the old line instead of the new
guarded one.

**Fix:** All three corrected -- comma added, typo fixed, `return` updated
to use the safe `raw_text` variable.

**Show:** Great real example for "syntax errors can point at a
misleading line number" and "always double check a fix actually made it
into the final merged file."

---

### 9. Speed vs. reliability tradeoff (concurrency tuning)

**What happened:** Runs were taking 90+ minutes with only 2 parallel
requests at a time. Raised to 6 for speed -- then later deliberately
lowered back to 2 after hitting a wave of errors likely caused by too
much concurrent load on certain models.

**Why:** More parallel requests = faster overall run, but also more load
hitting each model's endpoint at once, which can cause more timeouts/rate
limits. This is a genuine tradeoff, not a bug to "solve" perfectly.

**Fix:** No single right answer -- picked reliability over speed once
errors started stacking up, and documented *why* in a code comment rather
than leaving a confusing mismatch between the comment and the code.

**Show:** The `ThreadPoolExecutor(max_workers=...)` line in
`run_live_multi_model_eval.py`. Good one to explicitly frame as "a
deliberate engineering tradeoff I made on purpose," if asked about run
time.

---

### 10. Mistral: rate limiting (429) under concurrent load

**What happened:** Calls to `mistralai/mistral-large-3-675b-instruct-2512`
started returning `429 Too Many Requests`.

**Why:** Too many concurrent requests hitting that specific model's
endpoint at once.

**Fix:** Upgraded the retry logic to proper exponential backoff with
jitter (wait 2s, then 4s, then 8s, then 16s, each with a small random
offset) plus an extra cooldown specifically for confirmed 429s. Also
caught and fixed a regression introduced during this change: the new
version had accidentally started retrying *permanent* errors too (like a
wrong model name), instead of only retrying errors likely to succeed on
a second try.

**Show:** `nvidia_client.py`'s `_looks_transient()` function plus the
`sleep_duration = (BASE_RETRY_DELAY ** attempt) + ...` line. Good story
of "improved something, then caught my own regression before it caused
new problems."

---

### 11. Log files silently mixing old and new runs together

**What happened:** Early aggregation attempts risked combining leftover
rows from broken earlier runs (wrong model IDs, old bugs) with a clean
current run, since everything was appended to one shared log file.

**Why:** The original script always wrote to the same filename in
append mode, so every run's results piled on top of every previous run's
results, good and bad alike.

**Fix:** Switched to giving each run its own uniquely-named,
timestamped file (`run_20260705_1630.jsonl`, etc.), so every file
represents exactly one clean run with nothing else mixed in.

**Show:** `OUTPUT_LOG` in `run_live_multi_model_eval.py`, and
`get_latest_log_file()` in `aggregate_results.py` / `patch_eval.py`,
which automatically finds the most recent one.

---

### 12. `patch_eval.py`: retried items silently disappearing

**What happened:** A script built to retry only the failed rows had a
subtle bug: if a retried item failed *again*, it returned `None` instead
of a real row -- and since the script had already deleted the original
error row before retrying, that item ended up nowhere in the data at
all. Not logged as a success, not logged as an error -- just gone.

**Why:** `None` is "falsy" in Python, so `if result:` silently skipped
writing anything for that case.

**Fix:** Rewrote so every code path -- success or failure -- always
returns a real dictionary row. Also simplified the script overall: one
pass over the log file instead of two, removed a variable that was built
but never used, added an automatic backup file before making any
destructive changes.

**Show:** `call_and_score()` in `patch_eval.py` -- point out both
`return` statements (success and exception) have the *exact same set of
keys*, which is what guarantees nothing downstream breaks on a missing
field.

---

### 13. Aggregate results: hard to actually compare at a glance

**What happened:** `aggregate_results.py` produced a technically-correct
but awkward-to-read CSV -- one row per (model, axis) combination, in
"long format."

**Why:** Long format is great for further filtering/analysis, but bad for
visually comparing models side by side -- which is literally what
`SPEC.md`'s "comparison grid" deliverable asks for.

**Fix:** Added a second output: a pivoted table (models as rows, axes as
columns), reusing logic that already existed in the notebook but had
never been saved to a file.

**Show:** `build_comparison_grid()` in `aggregate_results.py`, and the
resulting `comparison_grid.csv`.

---

### 14. Llama-3.2-1b-instruct: persistent, unresolved endpoint failure

**What happened:** Every single attempt to call this model -- across at
least 4 separate runs, spanning many hours, including one final isolated
single-call test done specifically to check if it had recovered --
returned `Error code: 504`. It never once succeeded.

**Why:** Unlike the other issues on this list, this one was never a code
bug. Every other model worked once its specific issue was fixed; this
one simply never responded successfully, under any configuration,
including with full retry logic active.

**Decision:** Rather than keep re-running it indefinitely, documented it
as a known limitation and deferred it -- a specific, evidenced call
about when to stop debugging something that isn't actually a bug in the
code.

**Show:** No code to show for this one -- this is a genuinely good
verbal-only talking point: "not everything is a bug in your own code;
recognizing a real external outage, and knowing when to stop chasing it,
is part of the job too."

---

## If you want a short version to actually say out loud

"Building this pipeline surfaced a real mix of problems: some were typos
and simple bugs (wrong model IDs, a missing comma), some were about
different providers handling the same request differently (temperature
validation), some were about handling unreliable network calls gracefully
(retries, timeouts, rate limits), and one is a genuine unresolved
external issue I've chosen to document rather than keep chasing. Most of
today's actual debugging time went into integration issues with live
external APIs, not logic bugs in the code I originally designed."
