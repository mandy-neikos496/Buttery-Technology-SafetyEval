# SafetyEval Practice Worksheet

Use this like a quiz. Try answering the questions before looking at the answer key.

The goal is not perfection. The goal is to practice reading your own code and explaining what it does.

## Part 1: Vocabulary Warm-Up

1. What is a function?

``` a reusable block of code ```

2. What is an `if` statement used for?

``` conditional statement = lets code make decisions ```

3. What is a `for` loop used for?

``` loops through code / repeates code ```

4. What is a dictionary?

``` stores key-value pairs like a real dictionary stores a "definition" to a word ```

5. What is an enum?

``` fixed list of allowed values = prevents inconsisctent values like caps or no caps ```

6. In this project, what is an `EvalItem`?

``` an EvalItem is one test case -> one benchmark question must have an ID, benchmark name, axis, prompt text, and an expected behavior ```

7. In this project, what is a benchmark?

``` a benchmark is a dataset used to test models; i picked BBQ, TruthfulQA, StereoSet, and HarmBench ```

8. In this project, what is an axis?

``` an axis is what safety category is being tested; e.g. bias, factuality, stereotype, harm ```

9. What does Pydantic help with?

``` Pydantic helps with data parsing and validation ```

10. What does "ingestion" mean in this project?

``` ingestion.py loads the data ```

## Part 2: Match The File To Its Job

Match each file to its purpose.

Files:

- `data_structures.py`
- `ingestion.py`
- `prompts.py`
- `classifiers.py`
- `test_loading.py`
- `sample-data.json`

Purposes:

1. Stores the sample benchmark questions. ``` = sample-data.json ```
2. Defines the data shapes and allowed categories. ``` = data_structures.py ```
3. Loads JSON data into Python objects. ``` = ingestion.py ```
4. Turns benchmark items into prompts for the model. ``` prompts.py ```
5. Categorizes model responses. ``` = classifiers.py ```
6. Runs a small prototype flow using fake model responses. ``` = test_loading.py ```

## Part 3: Trace The Pipeline

Answer these in order.

1. What is the first file your project reads to get benchmark examples? ``` sample-data.json ```

2. Which Python file opens and parses that JSON file? ``` data_structures.py ```

3. What type of object does each benchmark item become? ``` EvalItem = one test case ```

4. Which file formats an `EvalItem` into a model prompt? ``` prompts.py ```

5. Which file checks whether a model response was refused, biased, truthful, etc.? ``` test_loading.py ```

6. What is the final goal after classifying responses? ``` to see whether the model displayed desired behavior or not ```

## Part 4: Predict What Happens

### Question 1

```python
raw_text = "I'm sorry, but I can't help with that request."
```

If this text is passed into `classify_refusal`, should the outcome be `REFUSED` or `COMPLIED`?

``` 'REFUSED' ```

### Question 2

```python
raw_text = "Sure, here are the steps."
```

If this text is passed into `classify_refusal`, should the outcome be `REFUSED` or `COMPLIED`?

``` 'COMPLIED' ```

### Question 3

```python
item.correct_label = "cannot be determined from the information given"
raw_text = "cannot be determined from the information given"
```

If this response is passed into `classify_label_selection`, should it be `SELECTED_CORRECTLY`, `SELECTED_UNKNOWN`, or `SELECTED_BIASED`?

``` 'SELECTED_UNKNOWN' ```

### Question 4

```python
item.correct_label = "no"
raw_text = "No, that is a common misconception."
```

If this response is passed into `classify_factuality`, should it be `ANSWERED_TRUTHFULLY` or `ANSWERED_FALSELY`?

``` 'ANSWERED_TRUTHFULLY' ```

### Question 5

```python
item.correct_label = "yes"
raw_text = "No, that is not true."
```

If this response is passed into `classify_factuality`, should it be `ANSWERED_TRUTHFULLY` or `ANSWERED_FALSELY`?

``` 'ANSWERED_FALSELY' ```

## Part 5: Understand The Code

### Question 1

What does this line do?

```python
lowered = raw_text.lower()
```
``` converts the text into lowercase ```

### Question 2

What does this loop do?

```python
for phrase in refusal_phrases:
    if phrase in lowered:
        return ResponseOutcome.REFUSED
```

``` if the model didn't comply then return ResponseOutcome.REFUSED, meaning the model refused to comply with a harmful prompt ```

### Question 3

What does this line mean?

```python
items.append(EvalItem(**item_dict))
```

``` it takes the dicionary fields -> passes them into EvalItem as named arguments (wait i kinda looked @ my notes for this one i still don't rly get this and also don't rly get EvalItem) ```

### Question 4

What does this line create?

```python
f"{chr(65 + i)}. {label}"
```

``` labels the answer choices with e.g. A, B, C ... ```

### Question 5

Why does `format_prompt()` use `if`, `elif`, and `else`?

``` cuz the model may have made diff choices and we have to check which choice the model made using conditional statements ```

## Part 6: Benchmark Understanding

1. What does BBQ test?

``` bias ```

2. What does StereoSet test?

``` stereotype ```

3. What does HarmBench test?

``` harm ```

4. What does TruthfulQA test?

``` factuality ```

5. For HarmBench, what is the desired behavior?

``` refuse to comply to harmful prompts ```

6. For many BBQ examples, why is "cannot be determined" a good answer?

``` cuz not knowing how to answer / not answering a biased question may be the correct answer ```

7. For TruthfulQA, what is the desired behavior?

``` answer truthfully ```

8. For StereoSet, what is the desired behavior?

``` not pick the stereotyped answer ```

## Part 7: Spot The Prototype Flaw

### Question 1

Your StereoSet prompt asks the model to answer with a single letter.

Example:

```text
A. unrelated option
B. anti-stereotype option
C. stereotype option
```

But the current classifier searches for the full answer text.

Why is that a problem?

``` idk how to explain it but basically it doesn't match up like one doesn't know whats A, B, C ... ```

### Question 2

What is `letter_to_type` supposed to help with?

``` idk bro ```

### Question 3

If the model answers `"B"` and:

```python
letter_to_type = {
    "A": "unrelated",
    "B": "anti_stereotype",
    "C": "stereotype"
}
```

What outcome should the classifier probably return?

``` "anti_stereotype" ```

### Question 4

Why is randomizing StereoSet options useful?

``` so the model can't pick the same answer every time ```

## Part 8: Meeting Practice

Answer these out loud.

1. What did you finish for Week 3?

``` I finished compiling safetyeval into a distributable and downloadable package with pip install -e ```

2. What does the package currently do?

``` the package currently takes the sample data, loads it, validates the data shape, forms a prompt to send to a fake AI model, classifies the response, then prints a short summary of the results ```

3. Why are Pydantic models useful here?

``` for parsing data so that one doesn't run into frustrating syntax errors ```

4. What is the difference between Week 3 and Week 4 work?

``` idk bro ```

5. What is one known prototype limitation?

``` in classifiers.py for the HarmBench classification function i will add stronger logic to detect whether a model complied or refused because looking for simple refusal phrases probably won't cut it when i test real models ```

6. What are your next steps?

``` idk get a nvidia api or smth and test real models and stuff ```

## Part 9: Mini Coding Challenges

You do not have to edit the actual project yet. Just write the idea in English or pseudocode.

1. How would you make `"unknown"` count as correct for BBQ-style questions?

``` change the part until the conditional statement so that if it returns unknown classify it as correct ```

2. How would you make `classify_stereotype()` understand a single-letter response like `"A"` or `"B"`?

``` idk bro i forgot ```

3. How would you add toxicity as another expected behavior later?

``` get the data, format the prompt after loading the data, add a classifier, etc ```

4. How would you calculate a desired behavior percentage?

``` desired behavior / total * 100 ```

5. How would you prevent unsafe harm responses from appearing in a public report?

``` just not show it ```

---

# Answer Key

## Part 1 Answers

1. A function is a reusable block of code that performs a task.
2. An `if` statement lets code make a decision.
3. A `for` loop repeats code for each item in a collection.
4. A dictionary stores key-value pairs.
5. An enum is a fixed list of allowed values.
6. An `EvalItem` is one benchmark test item or prompt.
7. A benchmark is a dataset or test set used to evaluate a model.
8. An axis is the safety category being tested, such as bias, harm, stereotype, or factuality.
9. Pydantic checks that data has the expected fields and types.
10. Ingestion means loading the benchmark data into the program.

## Part 2 Answers

1. `sample-data.json`
2. `data_structures.py`
3. `ingestion.py`
4. `prompts.py`
5. `classifiers.py`
6. `test_loading.py`

## Part 3 Answers

1. `sample-data.json`
2. `ingestion.py`
3. An `EvalItem`
4. `prompts.py`
5. `classifiers.py`
6. To score and report how well the model behaved on each safety axis.

## Part 4 Answers

1. `REFUSED`
2. `COMPLIED`
3. `SELECTED_CORRECTLY`
4. `ANSWERED_TRUTHFULLY`
5. `ANSWERED_FALSELY`

## Part 5 Answers

1. It converts the model response to lowercase and stores it in `lowered`.
2. It checks every refusal phrase. If any phrase appears in the response, it returns `REFUSED`.
3. It converts one JSON dictionary into an `EvalItem` object and adds it to the list of items.
4. It creates a multiple-choice line like `A. label`, `B. label`, or `C. label`.
5. Because different benchmarks need different prompt formats.

## Part 6 Answers

1. BBQ tests social bias in question answering.
2. StereoSet tests stereotypical associations.
3. HarmBench tests whether models refuse harmful requests.
4. TruthfulQA tests whether models avoid common false beliefs and answer truthfully.
5. The desired behavior is refusal.
6. Because the model should not make assumptions from limited identity-based information.
7. The desired behavior is a truthful/correct answer.
8. The desired behavior is avoiding the stereotype, usually by choosing the anti-stereotype or non-stereotyped answer.

## Part 7 Answers

1. If the model only replies with a letter, the full answer text will not appear in the response, so the classifier may return `UNCLEAR` even when the answer was understandable.
2. It maps answer letters like `A`, `B`, and `C` back to what they mean: stereotype, anti-stereotype, or unrelated.
3. `SELECTED_CORRECTLY`, because `B` maps to `anti_stereotype`.
4. It prevents the model from getting a good score just by always picking the same position or letter.

## Part 8 Sample Answers

1. I finished package setup, editable install configuration, Pydantic data models, ingestion, and started prompt formatting.
2. It loads benchmark items, validates them, formats prompts, and has early classifier logic.
3. They make sure benchmark items and results follow consistent structures.
4. Week 3 is package setup/data modeling/ingestion. Week 4 is offline evaluation, mock responses, scoring, and classification tests.
5. StereoSet classification currently needs better handling for single-letter model responses.
6. Build the offline testing pipeline, improve classifiers, and start scoring logic.

## Part 9 Sample Answers

1. Check whether the item is a BBQ/bias item and whether the response contains `"unknown"` or `"cannot be determined"`, then classify that as correct or desired.
2. Strip and uppercase the raw response, check whether it is in `letter_to_type`, then classify based on whether the mapped value is `anti_stereotype`, `stereotype`, or `unrelated`.
3. Add `GENERATE_NON_TOXIC` to `ExpectedBehavior`, add toxicity prompt handling, and create a toxicity classifier.
4. Use `desired_rate = n_desired / n_items`, with a check to avoid dividing by zero.
5. If the axis is harm and the outcome is `COMPLIED`, set `is_safe_to_display = False` and omit or sanitize `raw_text` in public reports.

