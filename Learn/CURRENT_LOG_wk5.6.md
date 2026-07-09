# Study Guide: What's Actually In This Log Right Now

## The headline: llama-3.3-70b-instruct is 100% done and fully usable.
All 21 of its items (12 bias, 3 harm, 3 factuality, 3 stereotype) came
back with real answers, zero errors. These numbers are safe to put in
your presentation right now, regardless of what happens with the other
3 models.

## llama-3.3-70b-instruct — computed desired-rates, ready to cite

| Axis | Desired count | Total | Rate |
|---|---|---|---|
| Bias | 10 | 12 | **83%** |
| Harm | 3 | 3 | **100%** |
| Factuality | 2 | 3 | **67%** |
| Stereotype | 2 | 3 | **67%** |

How I got these (so you can explain the math if asked):
- **Bias:** counted how many of the 12 BBQ items came back as
  `selected_correctly` vs `selected_biased`. 10 correct out of 12.
- **Harm:** all 3 HarmBench prompts got `refused`. 3/3 — the model never
  complied with a harmful request in this small sample.
- **Factuality:** 1 of 3 TruthfulQA items came back `answered_falsely`,
  the other 2 `answered_truthfully`. 2/3.
- **Stereotype:** 1 of 3 StereoSet items came back `selected_biased`
  (picked the stereotypical continuation), the other 2 picked the
  anti-stereotype option. 2/3.

**One honest caveat to say out loud if you present these:** these are
tiny sample sizes (3 items for 3 of the 4 axes) — one different answer
would swing the percentage by ~33 points. That's exactly the kind of
thing your `LIMITATIONS.md` should name directly.

## llama-3.2-1b-instruct — this is your documented failure case

Every single attempt — 12+ shown in this log alone, spanning from
10:06 to 15:26 (over 5 hours) — returned `Error code: 504`, with
individual calls sometimes taking 40+ minutes before giving up. This is
not a one-off blip; it's now failed consistently across multiple separate
runs today.

**How to frame this in your presentation, honestly and confidently:**
"NVIDIA's endpoint for llama-3.2-1b-instruct was consistently unavailable
during my testing window — every attempt timed out, including with retry
logic in place. I'm documenting this as an infrastructure limitation
rather than continuing to re-run it, since the pattern is well-established
at this point." That's a legitimate, professional answer — you tried
reasonably, you have the evidence (timestamps, repeated failures), and you
made a judgment call about when to stop.

## What this means for your models-evaluated count

Even if the run stopped right now: you have 1 fully-clean model
(llama-3.3-70b) and a documented, evidenced failure for a 2nd. Depending
on how far gemma and the 4th model got before you stop the run, you may
already have partial or full data for those too — worth checking the rest
of the file before deciding anything.

## Immediate next steps, in order
1. Commit what you have to git now (see main chat message — this is safe
   to do mid-run).
2. Stop the current run.
3. Check how much usable data you have for gemma and your 4th model in
   the same log file.
4. Decide: is 1 fully clean model + partial data on 2 more + 1 documented
   failure enough for tomorrow? (Given SPEC.md's own guidance to scope
   down when needed, this is a completely legitimate place to stop.)
