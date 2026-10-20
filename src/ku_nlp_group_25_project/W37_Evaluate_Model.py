"""Week 37 BIO span-prediction evaluation following the shared protocol.

Run from the repository root:
  PYTHONPATH=src uv run python -m ku_nlp_group_25_project.W37_Evaluate_Model --mode baseline
  PYTHONPATH=src uv run python -m ku_nlp_group_25_project.W37_Evaluate_Model --mode model

A predicted answer is a character span (start, end) in the FULL context, or None.
No gold annotations are used when producing model predictions. All validation
examples are scored once, including examples whose gold answer is truncated.
"""

import argparse
import re
import string
from collections import Counter

from ku_nlp_group_25_project.utilities.load_dataset import load_dataset

LANGUAGES = ("ar", "ko", "te")
MODEL_DIR = "models/w37_xlmr_bio_multilingual"
MAX_LENGTH = 384
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def gold_span(row):
    """Return the gold character span, or None for unanswerable examples."""
    if not bool(row["answerable"]):
        return None
    context, answer = str(row["context"]), str(row["answer"])
    start = int(row["answer_start"])
    end = start + len(answer)
    if not answer or start < 0 or context[start:end] != answer:
        raise ValueError(f"Invalid gold answer span at {start}: {answer!r}")
    return (start, end)


def answer_token_indices(context, span):
    """Context-only token indices whose offsets overlap the character span.

    Tokenization uses the original Week 37 regex tokenizer, applied to the
    complete context. Crucially, tokens are NOT split using gold answer bounds
    and prediction-side tokenization is identical to gold-side tokenization.
    """
    if span is None:
        return set()
    start, end = span
    return {
        index
        for index, match in enumerate(TOKEN_PATTERN.finditer(context))
        if match.start() < end and match.end() > start
    }


def span_token_counts(context, predicted_span, true_span):
    predicted = answer_token_indices(context, predicted_span)
    gold = answer_token_indices(context, true_span)
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def normalize_answer(text):
    # Standard SQuAD-style normalization for OPTIONAL integrated-QA metrics:
    # lowercase, ASCII punctuation stripping, English article stripping, spaces.
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_token_f1(prediction, reference):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(reference).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * p * r / (p + r)


def add_record(results, row, prediction):
    context = str(row["context"])
    true_span = gold_span(row)
    if prediction is not None:
        start, end = prediction
        if not (0 <= start < end <= len(context)):
            raise ValueError(f"Predicted span outside the context: {prediction}")

    tp, fp, fn = span_token_counts(context, prediction, true_span)
    sample_f1 = (1.0 if prediction is None and true_span is None
                 else prf(tp, fp, fn)[2])
    gold_text = context[true_span[0]:true_span[1]] if true_span else ""
    pred_text = context[prediction[0]:prediction[1]] if prediction else ""

    results.append({
        "lang": str(row["lang"]),
        "gold_answerable": true_span is not None,
        "pred_answerable": prediction is not None,
        "span_em": float(prediction == true_span),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "answer_token_example_f1": sample_f1,
        "squad_em": float(normalize_answer(pred_text) == normalize_answer(gold_text)),
        "squad_token_f1": qa_token_f1(pred_text, gold_text),
    })


def print_metrics(label, records):
    n = len(records)
    if not n:
        raise AssertionError(f"No evaluation records for {label}")

    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    tok_p, tok_r, tok_f1 = prf(tp, fp, fn)

    # Two mutually exclusive classes: answerable and unanswerable.
    by_class = {}
    for positive in (True, False):
        cls_tp = sum(r["pred_answerable"] == positive and
                     r["gold_answerable"] == positive for r in records)
        cls_fp = sum(r["pred_answerable"] == positive and
                     r["gold_answerable"] != positive for r in records)
        cls_fn = sum(r["pred_answerable"] != positive and
                     r["gold_answerable"] == positive for r in records)
        by_class[positive] = prf(cls_tp, cls_fp, cls_fn)
    macro_f1 = (by_class[True][2] + by_class[False][2]) / 2

    gold_empty = sum(not r["gold_answerable"] for r in records)
    pred_empty = sum(not r["pred_answerable"] for r in records)
    correct_empty = sum(not r["pred_answerable"] and
                        not r["gold_answerable"] for r in records)
    missed_answers = sum(not r["pred_answerable"] and
                         r["gold_answerable"] for r in records)

    avg = lambda field: sum(r[field] for r in records) / n
    print(f"\n=== {label} (n={n}) ===")
    print(f"Answer tokens: precision={tok_p:.4f}, recall={tok_r:.4f}, "
          f"micro-F1={tok_f1:.4f} (TP={tp}, FP={fp}, FN={fn})")
    print(f"Mean per-example answer-token F1: {avg('answer_token_example_f1'):.4f}")
    print(f"Exact character-span match: {avg('span_em'):.4f}")
    print(f"Answerability macro-F1: {macro_f1:.4f}")
    for is_answerable, classname in [(True, "answerable"), (False, "unanswerable")]:
        p, r, f = by_class[is_answerable]
        print(f"  {classname}: precision={p:.4f}, recall={r:.4f}, F1={f:.4f}")
    print(f"No-answer: gold={gold_empty}, predicted={pred_empty}, "
          f"correct={correct_empty}, missed-answers={missed_answers}")
    print(f"Supplementary SQuAD-v2-style normalized EM={avg('squad_em'):.4f}, "
          f"text-token F1={avg('squad_token_f1'):.4f}")


def evaluate(data, predictor):
    results = []
    for index, (_, row) in enumerate(data.iterrows(), 1):
        prediction = predictor(str(row["question"]), str(row["context"]))
        add_record(results, row, prediction)
        if index % 100 == 0:
            print(f"Evaluated {index}/{len(data)}", flush=True)

    for lang in LANGUAGES:
        print_metrics(lang, [r for r in results if r["lang"] == lang])
    print_metrics("OVERALL", results)
    assert len(results) == len(data), "Some validation examples were omitted!"
    return results


def get_model_predictor(model_dir):
    # Imports and GPU initialization only happen in --mode model.
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    model.eval()
    label_to_id = model.config.label2id
    B = label_to_id.get("B-ANS", 1)
    I = label_to_id.get("I-ANS", 2)

    def predict(question, context):
        batch = tokenizer(
            question, context,
            truncation="only_second", padding="max_length",
            max_length=MAX_LENGTH, return_offsets_mapping=True,
            return_tensors="pt",
        )
        sequence_ids = batch.sequence_ids(0)
        offsets = batch.pop("offset_mapping")[0].tolist()
        input_tensors = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            logits = model(**input_tensors).logits[0].cpu()
        labels = logits.argmax(dim=-1).tolist()
        b_probs = logits.softmax(dim=-1)[:, B]
        indices = [i for i, sid in enumerate(sequence_ids)
                   if sid == 1 and offsets[i][1] > offsets[i][0]]

        # Frozen decoder: choose the candidate B-ANS with highest B probability,
        # then extend rightward through consecutive I-ANS labels. No B => empty.
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
        return best_span
    return predict


def self_test():
    from types import SimpleNamespace
    context = "New York City is busy."
    # Uses prediction and gold character offsets; no learned model needed.
    s = (0, 8)
    assert answer_token_indices(context, s) == {0, 1}
    assert span_token_counts(context, s, s) == (2, 0, 0)
    assert span_token_counts(context, None, s) == (0, 0, 2)
    assert span_token_counts(context, s, None) == (0, 2, 0)
    assert span_token_counts(context, None, None) == (0, 0, 0)
    assert prf(2, 0, 0) == (1.0, 1.0, 1.0)
    assert qa_token_f1("", "") == 1.0
    assert qa_token_f1("New York", "York") > 0.0
    row = {"context": context, "lang": "ko", "answerable": True,
           "answer_start": 0, "answer": "New York"}
    records = []
    add_record(records, row, (0, 8))
    assert records[0]["span_em"] == 1.0
    add_record(records, row, None)
    assert records[1]["answer_token_example_f1"] == 0.0
    empty = {"context": context, "lang": "ar", "answerable": False,
             "answer_start": -1, "answer": "no"}
    add_record(records, empty, None)
    assert records[-1]["span_em"] == 1.0
    assert records[-1]["answer_token_example_f1"] == 1.0
    print("Evaluation metric self-tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "model", "self-test"),
                        default="baseline")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    self_test()
    if args.mode == "self-test":
        return

    _, validation = load_dataset()
    validation = validation[validation["lang"].isin(LANGUAGES)].copy()
    assert set(validation["lang"]) == set(LANGUAGES)
    print(f"Validation examples: {len(validation)} (expected 1,155)")
    assert len(validation) == 1155, "Validation split changed; investigate before scoring"

    if args.mode == "baseline":
        print("Baseline: always predict unanswerable / all O")
        predictor = lambda question, context: None
    else:
        print(f"Model directory: {args.model_dir}")
        predictor = get_model_predictor(args.model_dir)
    evaluate(validation, predictor)


if __name__ == "__main__":
    main()
