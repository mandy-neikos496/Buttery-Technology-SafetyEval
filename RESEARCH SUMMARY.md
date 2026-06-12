# Research Summary 

## 1. Benchmark: BBQ (Bias Benchmark for Question Answering)
* **What it tests:** Social biases across 9 dimensions (gender, race, age, etc.).
* **How it works:** It shows multiple-choice questions with short stories. It checks if the AI model relies on stereotypes when information is missing, or if its biases cause it to answer incorrectly even if the facts are stated.

---

## 2. Benchmark: StereoSet 
* **What it tests:** Whether an AI model relies on harmful stereotypes across four domains: gender, race, profession, and religion.
* **How it works:** Evaluate pretrained language models by running them through a large-scale natural English dataset, scores their responses, and then tracks their performance versus other models using a standardized leaderboard with a hidden test set.

---

## 3. Benchmark: HarmBench
* **What it tests:** The risks associated with the malicious use of LLMs by acting as a standardized evaluation framework for automated red teaming.
* **How it works:** It conducts a large-scale comparison between several red teaming methods, target LLMs, and defenses to rigorously assess and codevelop attacks plus defensive robustness.

---

## 4. Benchmark: TruthfulQA
* **What it tests:** Whether a language model is truthful when answering questions, specifically measuring if larger model sizes increasingly imitate human false beliefs plus misconceptions.
* **How it works:** It evaluates models using 817 questions spanning 38 categories (such as health, law, and politics) that are specifically crafted to cause humans, and AI in this case, to answer incorrectly due to common misconceptions.
