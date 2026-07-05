import pandas as pd
from datasets import load_dataset

from safetyeval.data_structures import EvalItem, EthicsAxis, ExpectedBehavior

# Harm axis; HarmBench

HARMBENCH_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

def load_real_harmbench_items(limit=10):
    """Loads behavior prompts from HarmBench's own Github repo (MIT license)."""
    df = pd.read_csv(HARMBENCH_CSV)
    
    df = df[df["FunctionalCategory"] == "standard"]

    items = []
    for i, row in df.head(limit).iterrows():
        item = EvalItem(
            id=f"harmbench-real-{i}",
            benchmark="harmbench-real",
            axis=EthicsAxis.HARM,
            input_text=row["Behavior"],
            expected_behavior=ExpectedBehavior.REFUSE,
            metadata={
                "semantic_category": row.get("SemanticCategory"),
                "behavior_id": row.get("BehaviorID"),
            },
            sensitivity="high",
        )
        items.append(item)
    
    return items
        
# Factuality axis; TruthfulQA

def load_real_truthfulqa_items(limit=10):
        """Loads TruthfulQA's multiple_choice config: a question plus several answers where one is correct."""
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")

        items = []
        for i, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
        
            correct_index = labels.index(1)
            correct_label = choices[correct_index]
        
            item = EvalItem(
                 id=f"truthfulqa-real-{i}",
                 benchmark="truthfulqa-mc-real",
                 axis=EthicsAxis.FACTUALITY,
                 input_text=row["question"],
                 expected_behavior=ExpectedBehavior.ANSWER_FACTUALLY, 
                 labels=choices,
                 correct_label=correct_label,
                 sensitivity="low",
            )
            items.append(item)

        return items

# Stereotype axis; StereoSet

def load_real_stereoset_items(limit=10):
    """
    Loads StereoSet's intersentence config: a context sentence plus three possible 
    continuations, each gold-labeled stereotype / anti-stereotype / unrelated.
    """
    dataset = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation", keep_in_memory=True)

    GOLD_LABEL_NAMES = {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}
    
    items = []
    sub_dataset = dataset.select(range(min(limit * 2, len(dataset))))

    for i, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        sentences = row["sentences"]["sentence"]
        gold_labels = row["sentences"]["gold_label"]

        completion_options = {}
        for sentence, gold_label in zip(sentences, gold_labels):
            label_name = GOLD_LABEL_NAMES.get(gold_label, "unrelated")
            key = label_name.replace("-", "_")
            completion_options[key] = sentence

        # Skip any row that doesn't have all 3 categories present.
        if not all(k in completion_options for k in ("stereotype", "anti_stereotype", "unrelated")):
            continue
            
        item = EvalItem(
             id=f"stereoset-real-{i}",
             benchmark="stereoset-real",
             axis=EthicsAxis.STEREOTYPE,
             input_text=row["context"],
             expected_behavior=ExpectedBehavior.AVOID_STEREOTYPE,
             completion_options=completion_options,
             metadata={"bias_type": row.get("bias_type"), "target": row.get("target")},
             sensitivity="medium",
        )
        items.append(item)
    
    return items

# Bias axis; BBQ
def load_real_bbq_items(limit_per_category=1):
    """Loads BBQ items directly from the heegyu/bbq dataset files on Hugging Face (CC BY 4.0)."""
    categories = [
        "Age",
        "Gender_identity",
        "Race_ethnicity",
        "Religion",
    ]

    items = []

    data_files = {cat: f"https://huggingface.co/datasets/heegyu/bbq/resolve/main/data/{cat}.jsonl" for cat in categories}

    for category in categories:
        dataset = load_dataset("json", data_files=data_files[category], split="train", keep_in_memory=True)
        n_rows = min(limit_per_category, len(dataset))

        for i, row in enumerate(dataset.select(range(n_rows))):
            labels = [row["ans0"], row["ans1"], row["ans2"]]
            correct_label = labels[int(row["label"])]

            item = EvalItem(
                id = f"bbq-{category.lower()}-real-{i}",
                benchmark = f"bbq-real-{category.lower()}",
                axis = EthicsAxis.BIAS,
                input_text = f"{row['context']}\n\n{row['question']}",
                expected_behavior = ExpectedBehavior.SELECT_LABEL,
                labels = labels,
                correct_label = correct_label,
                metadata = {
                    "category": category,
                    "context_condition": row.get("context_condition"),
                    "question_polarity": row.get("question_polarity"),
                },
                sensitivity = "medium",                          
            )

            items.append(item)

    return items