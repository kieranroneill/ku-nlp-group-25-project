
import re
import string
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from ku_nlp_group_25_project.utilities.load_dataset import load_dataset

MODEL_DIR = "models/w37_xlmr_qa"
MAX_LENGTH = 384

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_DIR
).to(device)
model.eval()

_, validation = load_dataset()
validation = validation[validation["lang"] == "ko"].copy()


def normalize_answer(text):
    text = text.lower()
    text = "".join(
        c for c in text if c not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction, gold):
    predicted = normalize_answer(prediction).split()
    reference = normalize_answer(gold).split()

    if not predicted or not reference:
        return float(predicted == reference)

    common = Counter(predicted) & Counter(reference)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(reference)

    return 2 * precision * recall / (precision + recall)


def predict_answer(question, context):
    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_LENGTH,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)

    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0].cpu()
    end_logits = outputs.end_logits[0].cpu()

    cls_index = encoded["input_ids"][0].tolist().index(
        tokenizer.cls_token_id
    )

    # Compare candidate answers with the no-answer score.
    null_score = (
        start_logits[cls_index] + end_logits[cls_index]
    ).item()

    context_indices = [
        i for i, sid in enumerate(sequence_ids)
        if sid == 1 and offsets[i][1] > offsets[i][0]
    ]

    best_score = float("-inf")
    best_answer = ""

    for start in context_indices:
        for end in context_indices:
            if end < start:
                continue
            if end - start + 1 > 30:
                continue

            score = (
                start_logits[start] + end_logits[end]
            ).item()

            if score > best_score:
                best_score = score
                best_answer = context[
                    offsets[start][0]:offsets[end][1]
                ]

    if null_score >= best_score:
        return ""

    return best_answer


exact_matches = []
f1_scores = []

for index, (_, row) in enumerate(validation.iterrows(), 1):
    prediction = predict_answer(
        row["question"], row["context"]
    )

    gold = row["answer"] if bool(row["answerable"]) else ""

    em = float(
        normalize_answer(prediction) == normalize_answer(gold)
    )
    f1 = f1_score(prediction, gold)

    exact_matches.append(em)
    f1_scores.append(f1)

    if index <= 5:
        print(f"\nExample {index}")
        print("Gold:", repr(gold))
        print("Predicted:", repr(prediction))

    if index % 50 == 0:
        print(f"Evaluated {index}/{len(validation)}", flush=True)

print("\n=== VALIDATION RESULTS ===")
print(f"Examples: {len(validation)}")
print(f"Exact Match: {100 * sum(exact_matches) / len(exact_matches):.2f}%")
print(f"Token F1: {100 * sum(f1_scores) / len(f1_scores):.2f}%")
