# SafetyEval Limitations

## Small Sample Size

SafetyEval evaluates 70 items per model: 40 BBQ items and 10 items each from HarmBench, TruthfulQA, and StereoSet. With only 10 items on three axes, one response can dramatically affect a model's percentage. The results are exploratory and should not be treated as sweeping statistical claims about a model's overall safety.

The project also compares solely four models. The models differ in size, architecture, training, and developer, so the results cannot prove that a single factor caused a score difference. Comparisons between developers based in the United States and international developers are descriptive only.

## Fixed Prompting Strategy

The evaluation uses one consistent zero-shot prompting style. Different instructions, examples, system prompts, or prompt wording may change a model's response. Therefore, the results describe model behavior under this specific evaluation setup as opposed to every possible interaction.

## Classifier Limitations

The regex HarmBench classifier detects direct refusal phrases but may miss polite or indirect refusals. When a refusal does not match one of its patterns, the classifier may incorrectly categorize the response as compliance.

The secondary LLM judge can understand more varied language, but it is not an objective source of truth. Its decisions may be influenced by its own biases, model version, and judge prompt. It also increases evaluation time and cost. When the judge fails or returns `UNCLEAR`, the CLI falls back to regex, meaning harm results may use different scoring sources.

## Benchmark and Scoring Scope

The four benchmarks cover bias, harm, factuality, and stereotype behavior, but they do not measure every area of AI safety. Benchmark performance may also not represent behavior in real-world interactions.

Multiple choice classifiers expect an obvious answer letter. A longer answer without a usable letter may be marked `UNCLEAR`, even if its meaning suggests a choice. API errors are excluded from scoring because they do not represent model behavior.

SafetyEval reports each axis separately instead of creating one overall safety score. Strong performance on one axis does not mean that a model is safe in every single area.

## Reproducibility and External Services

StereoSet answer order is randomized without a fixed seed, and the nearly deterministic temperature of `0.0001` does not actually guarantee identical responses. Model endpoints may also be changed or removed by their provider.

Since the project relies on NVIDIA Build, results can be affected by network problems, rate limits, API availability, and model updates.

## Report Safeguards

Raw HarmBench responses are hidden in order to avoid publishing potentially harmful content. This protects readers, but it also limits their ability to manually verify every harm classification from the public report.

## Future Improvements

Future work could use larger samples, multiple prompting styles, fixed random seeds, more models, multiple judge models, benign refusal testing, and additional safety axes.