import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
CHALLENGE_PATH = ROOT / "challenge.json"
W39_SCRIPT = ROOT / "W39_Learned_Answerability.py"
QA_SCRIPT = ROOT / "W40-41_Integrated_Multilingual_QA.py"
BILSTM_CHECKPOINT = ROOT / "w39_outputs" / "model2_bilstm.pt"
QA_CHECKPOINT = ROOT / "w40_41_outputs" / "seed44_epochs6" / "best_model"
OUTPUT_DIR = ROOT / "w44_challenge_results"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_bilstm(w39, frame, checkpoint_path):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocab = saved["vocabulary"]
    config = saved["config"]
    threshold = float(saved["threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = w39.RecurrentAnswerabilityClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()

    dataset = w39.AnswerabilityDataset(frame, vocab)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=w39.collate_model2_batch,
        num_workers=0,
    )

    probabilities = []
    with torch.no_grad():
        for questions, contexts, q_lengths, c_lengths, _labels in loader:
            logits = model(
                questions.to(device), contexts.to(device), q_lengths, c_lengths
            )
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame({
        "id": frame["id"].to_numpy(),
        "bilstm_p_answerable": probabilities,
        "bilstm_predicted_answerable": predictions,
    })


def run_qa(qa, records, checkpoint_dir):
    prediction_inputs = [
        {"id": r["id"], "question": r["question"], "context": r["context"]}
        for r in records
    ]
    predictions = qa.predict_new_examples(
        prediction_inputs, checkpoint_dir=checkpoint_dir
    )
    pred_by_id = predictions.set_index("id").to_dict("index")
    ordered_predictions = [{"id": r["id"], **pred_by_id[r["id"]]} for r in records]

    gold_examples = Dataset.from_list([
        {
            "id": r["id"],
            "example_id": i,
            "lang": r["lang"],
            "question": r["question"],
            "context": r["context"],
            "answerable": r["answerable"],
            "answers": {
                "text": [r["answer"]] if r["answerable"] else [],
                "answer_start": [r["answer_start"]] if r["answerable"] else [],
            },
        }
        for i, r in enumerate(records)
    ])
    languages = sorted({r["lang"] for r in records})
    qa_metrics, qa_table = qa.evaluate_predictions(
        gold_examples, ordered_predictions, languages
    )
    detail = qa_table[[
        "id", "prediction_text", "predicted_answerable", "exact_match", "token_f1",
        "no_answer_score", "prediction_start", "prediction_end",
    ]].copy()
    detail = detail.rename(columns={
        "prediction_text": "qa_prediction_text",
        "predicted_answerable": "qa_predicted_answerable",
        "no_answer_score": "qa_no_answer_score",
        "prediction_start": "qa_prediction_start",
        "prediction_end": "qa_prediction_end",
    })
    detail["qa_em"] = 100 * detail.pop("exact_match")
    detail["qa_token_f1"] = 100 * detail.pop("token_f1")
    return detail, qa_metrics


def classification_metrics(frame, prediction_column):
    subsets = [("overall", frame)]
    subsets += [(f"lang={lang}", g) for lang, g in frame.groupby("lang", sort=True)]
    subsets += [
        (f"created_by={creator}", g)
        for creator, g in frame.groupby("created_by", sort=True)
    ]
    rows = []
    for group_name, group in subsets:
        gold = group["answerable"].astype(int)
        predictions = group[prediction_column].astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(
            gold, predictions, labels=[0, 1], zero_division=0
        )
        rows.append({
            "group": group_name,
            "n_records": len(group),
            "accuracy": accuracy_score(gold, predictions),
            "macro_f1": f1.mean(),
            "unanswerable_precision": precision[0],
            "unanswerable_recall": recall[0],
            "unanswerable_f1": f1[0],
            "answerable_precision": precision[1],
            "answerable_recall": recall[1],
            "answerable_f1": f1[1],
            "n_unanswerable": int(support[0]),
            "n_answerable": int(support[1]),
        })
    return pd.DataFrame(rows)


def paired_results(frame):
    rows = []
    for (creator, pair_id), group in frame.groupby(["created_by", "pair_id"], sort=False):
        a = group[group["answerable"]].iloc[0]
        u = group[~group["answerable"]].iloc[0]
        b_correct = int(a["bilstm_predicted_answerable"] == 1 and
                        u["bilstm_predicted_answerable"] == 0)
        qa_abstained = int(u["qa_predicted_answerable"] == 0)
        qa_both_class = int(a["qa_predicted_answerable"] == 1 and qa_abstained)
        qa_both_em = int(a["qa_em"] == 100 and u["qa_em"] == 100)
        rows.append({
            "pair_id": pair_id,
            "created_by": creator,
            "lang": a["lang"],
            "answerable_id": a["id"],
            "unanswerable_id": u["id"],
            "bilstm_answerable_prediction": int(a["bilstm_predicted_answerable"]),
            "bilstm_unanswerable_prediction": int(u["bilstm_predicted_answerable"]),
            "bilstm_both_correct": b_correct,
            "qa_answerable_prediction": a["qa_prediction_text"],
            "qa_unanswerable_prediction": u["qa_prediction_text"],
            "qa_answerable_em": a["qa_em"],
            "qa_answerable_f1": a["qa_token_f1"],
            "qa_unanswerable_em": u["qa_em"],
            "qa_unanswerable_f1": u["qa_token_f1"],
            "qa_correct_abstention": qa_abstained,
            "qa_both_answerability_correct": qa_both_class,
            "qa_both_em_correct": qa_both_em,
        })
    return pd.DataFrame(rows)


def summarise_pairs(pairs):
    summaries = []
    groups = [("overall", pairs)]
    groups += [(f"lang={lang}", g) for lang, g in pairs.groupby("lang", sort=True)]
    groups += [
        (f"created_by={creator}", g)
        for creator, g in pairs.groupby("created_by", sort=True)
    ]
    for label, group in groups:
        summaries.append({
            "group": label,
            "n_pairs": len(group),
            "bilstm_both_correct_count": int(group["bilstm_both_correct"].sum()),
            "bilstm_pair_success_rate": group["bilstm_both_correct"].mean(),
            "qa_both_answerability_correct_count": int(group["qa_both_answerability_correct"].sum()),
            "qa_pair_answerability_success_rate": group["qa_both_answerability_correct"].mean(),
            "qa_both_em_correct_count": int(group["qa_both_em_correct"].sum()),
            "qa_pair_em_success_rate": group["qa_both_em_correct"].mean(),
        })
    return pd.DataFrame(summaries)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    records = json.loads(CHALLENGE_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(records)

    w39 = load_module("w39_frozen_module", W39_SCRIPT)
    qa = load_module("w40_41_frozen_module", QA_SCRIPT)

    bilstm_details = run_bilstm(w39, frame, BILSTM_CHECKPOINT)
    qa_details, qa_metrics = run_qa(qa, records, QA_CHECKPOINT)

    frame = frame.merge(bilstm_details, on="id", how="left")
    frame = frame.merge(qa_details, on="id", how="left")
    frame["bilstm_correct"] = (
        frame["answerable"].astype(int) == frame["bilstm_predicted_answerable"]
    )
    frame["qa_answerability_correct"] = (
        frame["answerable"].astype(int) == frame["qa_predicted_answerable"]
    )

    bilstm_metrics = classification_metrics(frame, "bilstm_predicted_answerable")
    qa_class_metrics = classification_metrics(frame, "qa_predicted_answerable")
    pairs = paired_results(frame)
    pair_summary = summarise_pairs(pairs)

    creator_rows = []
    for creator, group in frame.groupby("created_by", sort=True):
        creator_rows.append({
            "group": f"created_by={creator}",
            "n_examples": len(group),
            "squad_v2_em": group["qa_em"].mean(),
            "squad_v2_f1": group["qa_token_f1"].mean(),
        })
    qa_metrics = pd.concat([qa_metrics, pd.DataFrame(creator_rows)], ignore_index=True)

    outputs = {
        "record_predictions.csv": frame,
        "w39_bilstm_metrics.csv": bilstm_metrics,
        "w40_41_qa_metrics.csv": qa_metrics,
        "w40_41_answerability_metrics.csv": qa_class_metrics,
        "paired_results.csv": pairs,
        "paired_summary.csv": pair_summary,
    }
    for filename, data in outputs.items():
        data.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

    print(bilstm_metrics[["group", "n_records", "accuracy", "macro_f1"]].to_string(index=False))
    print(qa_metrics[["group", "n_examples", "squad_v2_em", "squad_v2_f1"]].to_string(index=False))
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()
