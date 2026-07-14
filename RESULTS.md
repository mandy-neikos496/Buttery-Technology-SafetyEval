# SafetyEval Results Report

## Overview

SafetyEval compared four language models across bias, harm, factuality, and stereotype behavior. Each model was tested on 70 benchmark items, producing 280 scoreable evaluations.

> **Content warning:** HarmBench contains potentially harmful requests. Raw harmful responses are not displayed in this report.

## Results

| Model | Bias | Harm | Factuality | Stereotype |
|---|---:|---:|---:|---:|
| Llama 3.2 1B | 47.5% | 100.0% | 20.0% | 30.0% |
| Gemma 2 2B | 62.5% | 100.0% | 70.0% | 30.0% |
| Qwen 3.5 122B-A10B | 72.5% | 100.0% | 90.0% | 50.0% |
| Llama 3.3 70B | 80.0% | 100.0% | 90.0% | 60.0% |

Higher percentages represent a greater rate of the desired behavior for that axis. The axes are reported separately rather than combined into one overall safety score.

## Main Findings

- Llama 3.3 70B had the highest bias and stereotype scores.
- Llama 3.3 70B and Qwen 3.5 tied for the highest factuality score.
- All four models received a 100.0% LLM-judge harm refusal rate.
- StereoSet was challenging for every model, with a highest score of 60.0%.
- The larger models generally performed better on the non-harm axes, but the sample is too small to conclude that model size caused the difference.

## Harm-Classifier Comparison

HarmBench responses were evaluated by both a regex classifier and a separate LLM judge.

| Model | Regex | LLM judge |
|---|---:|---:|
| Llama 3.2 1B | 90.0% | 100.0% |
| Gemma 2 2B | 20.0% | 100.0% |
| Qwen 3.5 122B-A10B | 80.0% | 100.0% |
| Llama 3.3 70B | 70.0% | 100.0% |

The classifiers disagreed on 14 of 40 harm responses, or 35.0%. In every disagreement, the LLM judge recognized a refusal that the regex classifier categorized as compliance. This suggests that regex-only scoring missed polite or indirect refusals.

The final reports use a valid LLM-judge outcome for harm. If the judge result is missing or unclear, the system falls back to regex.

## Model Comparisons

Qwen was the only model from an international developer. It performed similarly to or better than the average of the three models from the developers based in the United States. However, one international model is not enough to determine whether developer region affects model behavior.

The results also display stronger non-harm scores among the larger models. Model size cannot be evaluated separately from differences in architecture, training data, model version, and alignment methods, so this pattern is descriptive rather than causal.

## Methodology

The evaluation used BBQ, HarmBench, TruthfulQA, and StereoSet with a consistent zero-shot prompting strategy.

The desired-behavior percentage was calculated as:

`desired outcomes / scoreable items × 100`

API errors were excluded because they do not represent model behavior.

## Limitations

The evaluation utilizes a small sample and one fixed prompting strategy. Three axes contain only 10 items per model, so one response can change a score by 10 percentage points.

The regex classifier may miss indirect refusals, while the LLM judge can introduce its own bias or inconsistency. These results should be treated as exploratory rather than complete measurements of model safety.

See [LIMITATIONS.md](LIMITATIONS.md) for additional details.

## Conclusion

Llama 3.3 70B produced the strongest results across the non-harm axes, with Qwen 3.5 also performing strongly. The harm analysis demonstrated that classifier choice can substantially affect reported scores.

Overall, the project shows why model evaluations should clearly document their benchmarks, prompts, classifiers, and limitations.

[View the full HTML report](Buttery_Tech_Inc/safetyeval/safetyeval_pkg/realeval/report.html)