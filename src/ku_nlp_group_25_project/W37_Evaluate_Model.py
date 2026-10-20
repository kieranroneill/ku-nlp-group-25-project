
import re
import string
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ku_nlp_group_25_project.utilities.load_dataset import load_dataset

MODEL_DIR = "models/w37_xlmr_bio"
MAX_LENGTH = 384

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_DIR
).to(device)
model.eval()

_, validation = load_dataset()
validation = validation[validation["lang"] == "ko"].copy()

print("Using device:", device)
print("Validation examples:", len(validation))


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
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded.pop("offset_mapping")[0].tolist()

    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits[0].cpu()
    predicted_labels = logits.argmax(dim=-1).tolist()
    probabilities = torch.softmax(logits, dim=-1)

    context_indices = [
        i for i, sid in enumerate(sequence_ids)
        if sid == 1 and offsets[i][1] > offsets[i][0]
    ]

    # Find B-ANS tokens and extend through following I-ANS.
    # If there are multiple spans, select the one with
    # the highest B-ANS probability.
    best_answer = ""
    best_score = -1.0

    for position, start_token in enumerate(context_indices):
        if predicted_labels[start_token] != 1:
            continue

        end_position = position

        while (
            end_position + 1 < len(context_indices)
            and predicted_labels[
                context_indices[end_position + 1]
            ] == 2
        ):
            end_position += 1

        end_token = context_indices[end_position]

        score = probabilities[start_token, 1].item()

        if score > best_score:
            best_score = score

            char_start = offsets[start_token][0]
            char_end = offsets[end_token][1]

            best_answer = context[char_start:char_end]

    # No B-ANS predicted means no answer.
    return best_answer


exact_matches = []
f1_scores = []
no_answer_predictions = 0

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

    if prediction == "":
        no_answer_predictions += 1

    if index <= 5:
        print(f"\nExample {index}")
        print("Gold:", repr(gold))
        print("Predicted:", repr(prediction))

    if index % 50 == 0:
        print(
            f"Evaluated {index}/{len(validation)}",
            flush=True,
        )


print("\n=== BIO VALIDATION RESULTS ===")
print(f"Examples: {len(validation)}")
print(
    f"Exact Match: "
    f"{100 * sum(exact_matches) / len(exact_matches):.2f}%"
)
print(
    f"Token F1: "
    f"{100 * sum(f1_scores) / len(f1_scores):.2f}%"
)
print(f"No-answer predictions: {no_answer_predictions}")
