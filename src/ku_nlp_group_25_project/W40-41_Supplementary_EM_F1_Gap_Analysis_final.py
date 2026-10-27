import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
QA_SCRIPT = ROOT / "W40-41_Integrated_Multilingual_QA.py"
CHECKPOINT_DIR = ROOT / "w40_41_outputs/seed42_epochs6/best_model"
OUTPUT_DIR = CHECKPOINT_DIR.parent / "em_f1_gap_analysis"

GROUPS = [
    "Exact match",
    "Partial overlap",
    "Same tokens, different order",
    "No overlap / incorrect abstention",
]


def load_qa_module(path):
    spec = importlib.util.spec_from_file_location("w40_41_qa", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(em, f1):
    if np.isclose(em, 100.0):
        assert np.isclose(f1, 100.0)
        return GROUPS[0]
    if np.isclose(f1, 100.0):
        return GROUPS[2]
    if f1 > 0.0:
        return GROUPS[1]
    return GROUPS[3]


def choose_five_partial_overlap(table, languages):
    """Pick up to 5 review candidates: one per language, then F1 extremes."""
    partial = table.loc[table["group"] == GROUPS[1]]
    selected = []

    # First, represent as many languages as possible with mid-F1 cases.
    for lang in languages:
        candidates = partial.loc[(partial["lang"] == lang) & (~partial.index.isin(selected))]
        if not candidates.empty:
            middle = candidates["F1"].median()
            selected.append((candidates["F1"] - middle).abs().idxmin())

    # Then include low and high F1 examples to broaden the error spectrum.
    for kind in ("min", "max"):
        if len(selected) >= 5:
            break
        remaining = partial.loc[~partial.index.isin(selected)]
        if remaining.empty:
            break
        selected.append(remaining["F1"].idxmin() if kind == "min" else remaining["F1"].idxmax())

    # If fewer than 5 have been selected, fill using remaining central cases.
    while len(selected) < min(5, len(partial)):
        remaining = partial.loc[~partial.index.isin(selected)]
        mid = remaining["F1"].median()
        selected.append((remaining["F1"] - mid).abs().idxmin())

    return table.loc[selected].sort_values(["lang", "F1"])


def main():
    if not QA_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing original QA code: {QA_SCRIPT}")
    settings_path = CHECKPOINT_DIR / "qa_settings.json"
    if not settings_path.is_file():
        raise FileNotFoundError(f"Missing saved checkpoint settings: {settings_path}")

    qa = load_qa_module(QA_SCRIPT)
    saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    config = dict(saved_settings["config"])
    config["max_train_examples"] = None
    config["max_validation_examples"] = None  # ensure full validation split

    validation = qa.load_project_data(config)["validation"]
    print(f"Validation examples: {len(validation)}")
    # The model needs only question, context and id; gold data remain separate.
    records = [
        {"id": ex["id"], "question": ex["question"], "context": ex["context"]}
        for ex in validation
    ]

    # Reload the frozen checkpoint and its saved no-answer threshold.
    predictions = qa.predict_new_examples(records, checkpoint_dir=CHECKPOINT_DIR)
    gold_by_id = {ex["id"]: ex for ex in validation}
    assert len(predictions) == len(validation)
    assert predictions["id"].is_unique
    assert set(predictions["id"]) == set(gold_by_id)

    rows = []
    for pred in predictions.to_dict("records"):
        ex = gold_by_id[pred["id"]]
        em_01, f1_01 = qa.score_against_references(
            pred["prediction_text"], ex["answers"]
        )
        em, f1 = em_01 * 100.0, f1_01 * 100.0
        rows.append({
            "id": ex["id"],
            "example_id": ex["example_id"],
            "lang": ex["lang"],
            "group": classify(em, f1),
            "EM": em,
            "F1": f1,
            "F1_minus_EM": f1 - em,
            "gold_answerable": bool(ex["answerable"]),
            "predicted_answerable": bool(pred["predicted_answerable"]),
            "question": ex["question"],
            "context": ex["context"],
            "gold_answer": ex["answers"]["text"][0] if ex["answers"]["text"] else "",
            "prediction_text": pred["prediction_text"],
            "prediction_start": pred["prediction_start"],
            "no_answer_score": pred["no_answer_score"],
        })

    details = pd.DataFrame(rows)
    n = len(details)
    overall_em = details["EM"].mean()
    overall_f1 = details["F1"].mean()
    overall_gap = overall_f1 - overall_em

    grouped = details.groupby("group").agg(
        n=("id", "size"),
        mean_EM=("EM", "mean"),
        mean_F1=("F1", "mean"),
        sum_F1_minus_EM=("F1_minus_EM", "sum"),
    ).reindex(GROUPS)
    grouped["n"] = grouped["n"].fillna(0).astype(int)
    grouped["sum_F1_minus_EM"] = grouped["sum_F1_minus_EM"].fillna(0.0)
    grouped["share_percent"] = 100 * grouped["n"] / n
    grouped["gap_contribution_pp"] = grouped["sum_F1_minus_EM"] / n
    grouped["share_of_total_gap_percent"] = (
        100 * grouped["gap_contribution_pp"] / overall_gap
        if overall_gap > 1e-9 else 0.0
    )
    grouped.index.name = "group"
    grouped = grouped.reset_index()

    # Check that group contributions exactly reconstruct the overall gap.
    assert np.isclose(grouped["n"].sum(), n)
    assert np.isclose(grouped["gap_contribution_pp"].sum(), overall_gap)
    assert np.isclose(
        details.loc[details["EM"] > 99.999999, "F1_minus_EM"].sum(), 0.0
    )

    language_counts = pd.crosstab(details["lang"], details["group"])
    language_counts = language_counts.reindex(
        index=config["languages"], columns=GROUPS, fill_value=0
    )
    reviewed = choose_five_partial_overlap(details, config["languages"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details.to_csv(OUTPUT_DIR / "all_validation_scored.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(OUTPUT_DIR / "group_contributions.csv", index=False, encoding="utf-8-sig")
    language_counts.to_csv(OUTPUT_DIR / "group_counts_by_language.csv", encoding="utf-8-sig")
    reviewed.to_csv(OUTPUT_DIR / "five_partial_overlap_examples.csv", index=False, encoding="utf-8-sig")

    print(f"\nOverall: EM={overall_em:.4f}, F1={overall_f1:.4f}, F1-EM={overall_gap:.4f} pp")
    print("\nContributions by group:")
    print(grouped.round(3).to_string(index=False))
    print("\nCounts by language:")
    print(language_counts.to_string())
    print("\nFive examples for manual review:")
    print(reviewed[["id", "lang", "F1", "gold_answer", "prediction_text"]].to_string(index=False))

    saved_metrics = CHECKPOINT_DIR.parent / "validation_metrics.csv"
    if saved_metrics.is_file():
        original = pd.read_csv(saved_metrics).set_index("group").loc["overall"]
        print("\nOriginal reported EM/F1:",
              f"{original['squad_v2_em']:.4f} / {original['squad_v2_f1']:.4f}")
        print("Recomputed EM/F1:       ", f"{overall_em:.4f} / {overall_f1:.4f}")
    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()