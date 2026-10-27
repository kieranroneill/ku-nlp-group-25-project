
import argparse
import csv
import json
import importlib.util
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATA = "data/challenge_sets/challenge.json"
DEFAULT_OUTPUT = "outputs/week_43_44/challenge_predictions.csv"
DEFAULT_SUMMARY = "outputs/week_43_44/challenge_metrics.csv"
DEFAULT_W37 = "models/w37_xlmr_bio_multilingual"


# Load and validate challenge records from the official JSON file.
def read_challenge(path):
    with Path(path).open(encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list) or not rows:
        raise ValueError("Challenge JSON must contain a non-empty list")

    required = {
        "id", "pair_id", "created_by", "question", "context",
        "lang", "answerable", "answer_start", "answer", "source_url"
    }

    ids = set()
    pairs = defaultdict(list)

    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Challenge record has missing required fields")

        if row["id"] in ids:
            raise ValueError(f"Duplicate ID: {row['id']}")

        ids.add(row["id"])
        pairs[row["pair_id"]].append(row)

        if row["lang"] not in {"ar", "ko", "te", "zh"}:
            raise ValueError(f"Unexpected language: {row['lang']}")

        if not isinstance(row["answerable"], bool):
            raise ValueError(f"Invalid answerability: {row['id']}")

        answer = row["answer"]
        start = row["answer_start"]
        context = row["context"]

        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"Invalid answer offset: {row['id']}")

        if row["answerable"]:
            if not answer or start < 0 or context[start:start + len(answer)] != answer:
                raise ValueError(f"Invalid answer span: {row['id']}")
        elif answer != "" or start != -1:
            raise ValueError(f"Invalid unanswerable record: {row['id']}")

    for pair_id, group in pairs.items():
        if len(group) != 2:
            raise ValueError(f"Invalid pair size: {pair_id}")

        if {row["answerable"] for row in group} != {True, False}:
            raise ValueError(f"Pair must contain one answerable and one unanswerable record: {pair_id}")

        for field in ("context", "lang", "created_by", "source_url"):
            if group[0][field] != group[1][field]:
                raise ValueError(f"Pair mismatch in {field}: {pair_id}")

    print(f"Validated {len(rows)} records in {len(pairs)} pairs.")
    return rows

# Normalize answer text for consistent scoring.
def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# Calculate exact match and token F1 for one prediction.
def qa_scores(predicted, gold):
    p, g = normalize_answer(predicted).split(), normalize_answer(gold).split()
    em = float(p == g)
    if not p or not g:
        return em, em
    overlap = sum((Counter(p) & Counter(g)).values())
    f1 = 2 * overlap / (len(p) + len(g))
    return em, f1


# Run the frozen W37 BIO model on all challenge examples.
def run_w37(rows, model_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"W37 model checkpoint not found: {model_dir}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"W37 device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()
    label_to_id = model.config.label2id
    B, I = label_to_id.get("B-ANS", 1), label_to_id.get("I-ANS", 2)
    predictions = []
    for number, row in enumerate(rows, start=1):
        context = row["context"]
        batch = tokenizer(row["question"], context,
                          truncation="only_second", padding="max_length",
                          max_length=384, return_offsets_mapping=True,
                          return_tensors="pt")
        sequence_ids = batch.sequence_ids(0)
        offsets = batch.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[0].cpu()
        labels = logits.argmax(dim=-1).tolist()
        b_probs = logits.softmax(dim=-1)[:, B]
        indices = [i for i, sid in enumerate(sequence_ids)
                   if sid == 1 and offsets[i][1] > offsets[i][0]]
        best_span, best_score = None, float("-inf")
        for pos, i in enumerate(indices):
            if labels[i] != B:
                continue
            last = pos
            while last + 1 < len(indices) and labels[indices[last + 1]] == I:
                last += 1
            score = b_probs[i].item()
            if score > best_score:
                best_score = score
                best_span = (offsets[i][0], offsets[indices[last]][1])
        predictions.append(context[best_span[0]:best_span[1]] if best_span else "")
        print(f"W37 evaluated {number}/{len(rows)}: {row['id']}")
    return predictions


# Run the frozen integrated QA model on all challenge examples.
def run_w40(rows, source_path, checkpoint_dir):
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"W40--41 Python module not found: {source}")
    if not (Path(checkpoint_dir) / "qa_settings.json").is_file():
        raise FileNotFoundError("Missing integrated model checkpoint/qa_settings.json")
    spec = importlib.util.spec_from_file_location("integrated_qa_challenge", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    examples = [{"id": row["id"], "example_id": i,
                 "lang": row["lang"], "question": row["question"],
                 "context": row["context"]} for i, row in enumerate(rows)]
    table = module.predict_new_examples(examples, checkpoint_dir=checkpoint_dir)
    if list(table["id"]) != [row["id"] for row in rows]:
        raise ValueError("Integrated QA prediction rows are not aligned with challenge IDs")
    return table["prediction_text"].fillna("").astype(str).tolist()


# Score each prediction against its reference answer.
def score_model(rows, model_name, predictions):
    records = []
    for row, text in zip(rows, predictions):
        text = str(text)
        em, f1 = qa_scores(text, row["answer"])
        records.append({**row, "model": model_name, "prediction": text,
                        "predicted_answerable": int(bool(text.strip())),
                        "exact_match": em, "token_f1": f1})
    return records


# Calculate model-level and paired-example performance metrics.
def summarize(records):
    summaries = []
    for model in sorted({r["model"] for r in records}):
        group = [r for r in records if r["model"] == model]
        pair_ids = sorted({r["pair_id"] for r in group})
        a = [r for r in group if int(r["answerable"]) == 1]
        u = [r for r in group if int(r["answerable"]) == 0]
        pair_success = sum(all(r["exact_match"] == 1 for r in group if r["pair_id"] == pair)
                           for pair in pair_ids)
        class_f1 = []
        for label in [0, 1]:
            tp = sum(int(r["answerable"]) == label and r["predicted_answerable"] == label for r in group)
            fp = sum(int(r["answerable"]) != label and r["predicted_answerable"] == label for r in group)
            fn = sum(int(r["answerable"]) == label and r["predicted_answerable"] != label for r in group)
            class_f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        summaries.append({
            "model": model, "n": len(group),
            "overall_em": sum(r["exact_match"] for r in group) / len(group),
            "overall_token_f1": sum(r["token_f1"] for r in group) / len(group),
            "answerable_em": sum(r["exact_match"] for r in a) / len(a),
            "answerable_token_f1": sum(r["token_f1"] for r in a) / len(a),
            "unanswerable_em": sum(r["exact_match"] for r in u) / len(u),
            "unanswerable_token_f1": sum(r["token_f1"] for r in u) / len(u),
            "answerability_macro_f1": sum(class_f1) / 2,
            "fully_correct_pairs": pair_success,
        })
    return summaries


# Save prediction or summary rows to a CSV file.
def write_csv(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {target}")


# Check the answer-scoring logic with simple examples.
def self_test():
    assert qa_scores("", "") == (1.0, 1.0)
    assert qa_scores("wrong", "") == (0.0, 0.0)
    assert qa_scores("Athens", "Athens") == (1.0, 1.0)
    assert qa_scores("in Athens", "Athens") == (0.0, 2 / 3)
    print("Scoring self-tests passed.")


# Validate the data, run the selected model, and save results.
def main():
    parser = argparse.ArgumentParser(description="Validate and evaluate the W43-W44 QA challenge set")
    parser.add_argument("--mode", choices=["validate", "w37", "w40", "both"], default="validate")
    parser.add_argument("--challenge", default=DEFAULT_DATA)
    parser.add_argument("--w37-model", default=DEFAULT_W37)
    parser.add_argument("--w40-script", default="src/ku_nlp_group_25_project/W40-41_Integrated_Multilingual_QA.py")
    parser.add_argument("--w40-checkpoint", default="w40_41_outputs/seed44_epochs6/best_model")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    self_test()
    rows = read_challenge(args.challenge)
    if args.mode == "validate":
        print("Validation only. No model was evaluated and no predictions were written.")
        return
    predictions = []
    if args.mode in {"w37", "both"}:
        predictions += score_model(rows, "W37_BIO", run_w37(rows, args.w37_model))
    if args.mode in {"w40", "both"}:
        predictions += score_model(rows, "W40_Integrated_QA", run_w40(rows, args.w40_script, args.w40_checkpoint))
    write_csv(args.output, predictions)
    summaries = summarize(predictions)
    write_csv(args.summary, summaries)


    # Count the total number of challenge pairs.
    n_pairs = len({r["pair_id"] for r in rows})

    for row in summaries:
        print(f"{row['model']}: EM={row['overall_em']:.3f}, token F1={row['overall_token_f1']:.3f}, "
              f"answerability macro-F1={row['answerability_macro_f1']:.3f}, "
              f"fully correct pairs={row['fully_correct_pairs']}/{n_pairs}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
