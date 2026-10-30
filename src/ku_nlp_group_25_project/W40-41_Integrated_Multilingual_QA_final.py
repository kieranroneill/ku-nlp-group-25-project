#  1. Imports and settings
import json
import math
import platform
import random
import re
import string
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# CONFIG collects experiment settings, including the Lab's base encoder, in one reusable dictionary.
CONFIG = {
    "model_name": "FacebookAI/xlm-roberta-base",
    "model_revision": "main",
    "data_path": r"D:\Python\nlp_data\data\tydi_xor_rc_ar_ko_te",
    "languages": ["ar", "ko", "te"],
    "seed": 44,
    "epochs": 6,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_length": 384,              # question + context + special tokens
    "max_question_tokens": 128,     # Truncate very long questions to leave room for the context.
    "stride": 128,                  # Number of overlapping context tokens between adjacent windows.
    "n_best_size": 20,
    "max_answer_length": 30,        # Maximum candidate answer length in subword tokens (following the Lab).
    "mixed_precision": True,
    "gradient_checkpointing": True, # Trade extra computation for lower GPU memory use; this does not save the best model checkpoint.
    "threshold_objective": "squad_v2_f1",  # Select one global threshold to maximise overall validation QA F1.
    "max_train_examples": None,
    "max_validation_examples": None,
    "output_dir": "w40_41_outputs/seed44_epochs6",
}

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def set_seed(seed):
    # [Lab 6 reused] Seed initialisation and shuffling; bitwise reproducibility across hardware is not guaranteed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#  2. Load the data and split
def load_project_data(config):
    raw = load_from_disk(config["data_path"])
    result = {}
    for split in ["train", "validation"]:
        data = raw[split].filter(lambda row: row["lang"] in config["languages"])
        # Assign row IDs after language filtering and before subsampling; add split prefixes to distinguish train and validation IDs.
        data = data.map(lambda row, idx: {
            "example_id": idx,
            "id": f"{split}-{idx}",
            # Original unanswerable answers may contain yes/no; represent them as empty answers for this QA task.
            "answers": {
                "text": [row["answer"]] if row["answerable"] else [],
                "answer_start": [int(row["answer_start"])] if row["answerable"] else [],
            },
        }, with_indices=True)
        limit_key = "max_train_examples" if split == "train" else "max_validation_examples"
        limit = config[limit_key]
        if limit is not None:
            data = data.shuffle(seed=config["seed"]).select(range(min(limit, len(data))))
        result[split] = data
    return DatasetDict(result)


#  3. Tokenization, windows, and character -> subword alignment
def prepare_features(examples, tokenizer, config, training):
    # [Lab 6 reused] Inputs are ordered (question, context); sequence ID 1 identifies context tokens.
    # Use the pretrained subword tokenizer, not split(); do not lowercase or alter the context, so offsets remain valid.
    questions = []
    for question in examples["question"]:
        q_offsets = tokenizer(question, add_special_tokens=False,
                              return_offsets_mapping=True)["offset_mapping"]
        if len(q_offsets) > config["max_question_tokens"]:
            # Truncate the question at a subword boundary.
            question = question[:q_offsets[config["max_question_tokens"] - 1][1]]
        questions.append(question)

    encoded = tokenizer(
        questions, examples["context"],
        max_length=config["max_length"], truncation="only_second",
        stride=config["stride"], padding="max_length",
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    # Process each window and prepare answer start/end token labels for training.
    # Convert character offsets into the two token-level labels: start_position and end_position.
    sample_map = encoded.pop("overflow_to_sample_mapping")
    example_ids, cls_indices = [], []
    starts, ends = [], []

    # i indexes the tokenised window; sample_index identifies its original example within the current input batch.
    for i, sample_index in enumerate(sample_map):
        example_ids.append(examples["id"][sample_index])
        # Locate CLS in this input; its position represents a window with no answer.
        cls_index = encoded["input_ids"][i].index(tokenizer.cls_token_id)
        cls_indices.append(cls_index)
        # Sequence IDs identify question tokens (0), context tokens (1), and special/padding tokens (None).
        sequence_ids = encoded.sequence_ids(i)
        # Read character offsets; context-token offsets refer to the original context string.
        offsets = encoded["offset_mapping"][i]
        # Collect context-token indices in the combined input so answer alignment stays within the context.
        context_indices = []
        for j, sid in enumerate(sequence_ids):
            if sid == 1:
                context_indices.append(j)

        if training:
            # Default to CLS/CLS (no answer in this window), then replace these labels if the full answer is present.
            # Keep CLS/CLS in either of two cases:
            # 1. The example is unanswerable. 2. It is answerable, but this window does not contain the complete answer.
            # These are window-level labels: different windows from one answerable example can receive different labels.
            start_position = end_position = cls_index
            answers = examples["answers"][sample_index]
            if answers["text"] and context_indices:
                # For an answerable example with context tokens in this window:
                # Read the answer's character span and the window's first and last context-token indices.
                # Check that the window starts no later than the answer and ends no earlier than the answer.
                # Scan from the left for the start token and from the right for the end token using the two while loops.
                # The end token index is inclusive; the character-span end is exclusive.
                # Store one start/end pair per window: the answer's token indices or CLS/CLS for no answer.
                char_start = answers["answer_start"][0]
                char_end = char_start + len(answers["text"][0])
                first, last = context_indices[0], context_indices[-1]
                if offsets[first][0] <= char_start and offsets[last][1] >= char_end:
                    left, right = first, last
                    while left <= last and offsets[left][0] <= char_start:
                        left += 1
                    while right >= first and offsets[right][1] >= char_end:
                        right -= 1
                    start_position, end_position = left - 1, right + 1
            starts.append(start_position)
            ends.append(end_position)

        # Keep offsets only for context tokens; mask question, special, and padding offsets for span decoding.
        new_offsets = []
        for j, offset in enumerate(offsets):
            if sequence_ids[j] == 1:
                new_offsets.append(offset)
            else:
                new_offsets.append(None)
        encoded["offset_mapping"][i] = new_offsets

    encoded["example_id"] = example_ids
    encoded["cls_index"] = cls_indices
    if training:
        encoded["start_positions"] = starts
        encoded["end_positions"] = ends
    return encoded

def collate_batch(rows):
    # [Lab 6 reused] Remove right padding that is unused by every sequence in the batch to reduce computation.
    width = max(sum(row["attention_mask"]) for row in rows)
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in ["input_ids", "attention_mask", "token_type_ids"]:
            values = [value[:width] for value in values]
        batch[key] = torch.tensor(values, dtype=torch.long)
    return batch

def make_loader(features, config, training):
# Build a DataLoader from the preprocessed windows.
# Choose model input columns, batch size, shuffling, and the batch collation function.
    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in features.column_names:
        columns.append("token_type_ids")
    if training:
        # During training, provide gold start/end token positions for loss computation.
        columns += ["start_positions", "end_positions"]
    # Do not pass metadata to the model; validation inputs exclude gold start/end labels.
    return DataLoader(
        # Select model input columns; keep other feature metadata separately for post-processing.
        features.select_columns(columns),
        # Use the configured training or evaluation batch size.
        batch_size=config["train_batch_size" if training else "eval_batch_size"],
        # Shuffle only during training and collate each batch into PyTorch tensors.
        shuffle=training, collate_fn=collate_batch,
    )

#  4. Training and inference
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, config):
    # [Lab 6 reused] Train over all windows once: forward pass, loss, backpropagation, and parameter updates; return mean loss.
    # The optimiser updates parameters using gradients; the learning rate controls the step size.
    # The scheduler changes the optimiser's learning rate during training.
    # The scaler helps maintain numerical stability in FP16 training.
    model.train()
    # Clear any gradients remaining from the previous epoch.
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_features = 0.0, 0
    # Accumulate this many batches before each parameter update.
    accumulation = config["gradient_accumulation_steps"]
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        for key, value in batch.items():
            batch[key] = value.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            # Run the forward pass and compute the loss.
            loss = model(**batch).loss
        batch_size = batch["input_ids"].shape[0]
        # Accumulate batch mean loss times the number of windows to obtain a window-weighted epoch mean.
        total_loss += loss.item() * batch_size
        total_features += batch_size
        # [Added] Backpropagate while accumulating gradients.
        # With 5 batches and accumulation=2, the final group has one batch, so divide its loss by 1 rather than 2.
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        scaler.scale(loss / group_size).backward()
        # Update parameters once the accumulation group is complete, including the final partial group.
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # If FP16 overflow skips an optimiser update, do not advance the scheduler either.
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            # Advance the scheduler only after a successful update; it sets the learning rate for subsequent updates.
            optimizer.zero_grad(set_to_none=True)
    # Return mean loss measured during training; parameters have already been updated for validation in main().
    return total_loss / total_features

@torch.no_grad()
def predict_logits(model, loader, device):
    # [Lab 6 reused] Prediction differs from training in the following ways:
    # Training dropout randomly zeros activations; evaluation disables this randomness.
    # Training tracks a computation graph and the information needed to calculate gradients.
    # Prediction needs no backpropagation, so no_grad disables gradient tracking in this function.
    model.eval()
    start_logits, end_logits = [], []
    for batch in tqdm(loader, desc="Predicting"):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        start_logits.extend(outputs.start_logits.float().cpu().numpy())
        end_logits.extend(outputs.end_logits.float().cpu().numpy())
    return start_logits, end_logits


#  5. Candidate spans and no-answer scores
def decode_candidates(examples, features, logits, config):
    # [Lab 6 reused] Aggregate windows for each original example and search for valid answer spans.
    # Group windows, examine candidates, discard invalid spans, select the best span, and calculate the no-answer score.
    by_example = defaultdict(list)
    for i, example_id in enumerate(features["example_id"]):
        by_example[example_id].append(i)
    start_logits, end_logits = logits
    candidates = []
    for example in examples:
        best_score, null_score = -float("inf"), float("inf")
        best_text, best_start, best_end = "", -1, -1
        for feature_index in by_example[example["id"]]:
            feature = features[feature_index]
            starts, ends = start_logits[feature_index], end_logits[feature_index]
            offsets = feature["offset_mapping"]
            cls = feature["cls_index"]
            # [Added] Score the null answer as well as candidate spans so decoding can choose whether to answer.
            # null score = start logit at CLS + end logit at CLS.
            # span score = start logit at the candidate start + end logit at the candidate end.
            # Use the minimum null score across windows and the maximum valid span score, following common HF post-processing.
            null_score = min(null_score, float(starts[cls] + ends[cls]))
            valid = np.array([j for j in range(len(starts))
                              if offsets[j] is not None and offsets[j][1] > offsets[j][0]])
            start_choices = valid[np.argsort(starts[valid])[::-1][:config["n_best_size"]]]
            end_choices = valid[np.argsort(ends[valid])[::-1][:config["n_best_size"]]]
            for start in start_choices:
                for end in end_choices:
                    if end < start or end - start + 1 > config["max_answer_length"]:
                        continue
                    score = float(starts[start] + ends[end])
                    if score > best_score:
                        best_score = score
                        best_start, best_end = offsets[start][0], offsets[end][1]
                        # Slice the original context to preserve spacing and punctuation; do not reconstruct answers with tokenizer.decode.
                        best_text = example["context"][best_start:best_end]
        candidates.append({
            "id": example["id"], "best_span_text": best_text,
            "best_span_start": best_start, "best_span_end": best_end,
            "best_span_score": best_score, "null_score": null_score,
            # A larger null-minus-span score favours no answer; this logit difference is not a calibrated probability.
            "no_answer_score": null_score - best_score,
        })
    # Store the best span and scores; the validation-selected threshold later determines whether to return text or an empty answer.
    return candidates

#  6. SQuAD-v2 scoring and threshold selection
# Score answers, select a validation threshold, form final predictions, and summarise metrics; neural parameters are unchanged.
# [Lab 6 adapted] Extend the first three scoring functions to handle empty answers; add threshold selection and combined evaluation.
def normalise_answer(text):
# Normalise English answer strings for comparison.
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def score_text(prediction, gold):
# Compare one prediction with one reference and return EM and token F1.
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    exact = float(pred_tokens == gold_tokens)
    if not pred_tokens or not gold_tokens:
        return exact, float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    f1 = 2.0 * overlap / (len(pred_tokens) + len(gold_tokens))
    return exact, f1

def score_against_references(prediction, answers):
# Handle examples with multiple reference answers.
    references = [text for text in answers["text"] if normalise_answer(text)] or [""]
    scores = [score_text(prediction, gold) for gold in references]
    return max(score[0] for score in scores), max(score[1] for score in scores)

def select_threshold(examples, candidates):
# [Added] Use validation gold answers to select one global threshold maximising mean QA token F1.
    # Threshold sweep:
    # 1. Precompute each example's F1 for an empty prediction and for its best span.
    # 2. Start with all predictions empty, setting the threshold below every finite score difference.
    # 3. Raise the threshold so examples switch from empty predictions to their best spans.
    # 4. Retain the threshold with the highest mean F1.
    deltas = np.array([row["no_answer_score"] for row in candidates], dtype=float)
    empty_f1 = np.array([score_against_references("", ex["answers"])[1] for ex in examples])
    span_f1 = np.array([score_against_references(row["best_span_text"], ex["answers"])[1]
                        for ex, row in zip(examples, candidates)])
    finite = deltas[np.isfinite(deltas)]
    # Examples without a valid candidate remain empty even when the threshold increases.
    initial_threshold = float(np.nextafter(finite.min(), -np.inf)) if len(finite) else 0.0
    total = float(empty_f1.sum())
    best_total, best_threshold = total, initial_threshold
    curve = [{"threshold": initial_threshold, "squad_v2_f1": 100 * total / len(examples)}]
    for threshold in np.unique(finite):
        switched = deltas == threshold
        total += float((span_f1[switched] - empty_f1[switched]).sum())
        curve.append({"threshold": float(threshold), "squad_v2_f1": 100 * total / len(examples)})
        if total > best_total + 1e-10:
            best_total, best_threshold = total, float(threshold)
    return best_threshold, pd.DataFrame(curve)

def apply_threshold(candidates, threshold):
# Apply the selected threshold to generate final answers.
    predictions = []
    for row in candidates:
        no_answer = row["no_answer_score"] > threshold or row["best_span_text"] == ""
        predictions.append({
            **row,
            "prediction_text": "" if no_answer else row["best_span_text"],
            "prediction_start": -1 if no_answer else row["best_span_start"],
            "prediction_end": -1 if no_answer else row["best_span_end"],  # exclusive
            "predicted_answerable": int(not no_answer),
        })
    return predictions

def evaluate_predictions(examples, predictions, languages):
# [Added] Build per-example predictions and aggregate evaluation tables.
    # Evaluate both answer-text quality and the decision about whether to answer.
    # Return overall/per-language metrics for reporting and prediction details for manual error analysis.
    records = []
    for ex, pred in zip(examples, predictions):
        em, f1 = score_against_references(pred["prediction_text"], ex["answers"])
        records.append({
            "example_id": ex["example_id"], "lang": ex["lang"],
            "question": ex["question"], "context": ex["context"],
            "gold_answerable": int(ex["answerable"]),
            "gold_answer": ex["answers"]["text"][0] if ex["answers"]["text"] else "",
            "gold_start": ex["answers"]["answer_start"][0] if ex["answers"]["text"] else -1,
            **pred, "exact_match": em, "token_f1": f1,
            "answerability_correct": int(ex["answerable"]) == pred["predicted_answerable"],
        })
    table = pd.DataFrame(records)
    results = []
    for group in ["overall"] + list(languages):
        subset = table if group == "overall" else table[table["lang"] == group]
        if len(subset) == 0:  # A language may be absent in a small smoke-test subset.
            continue
        precision, recall, f1, support = precision_recall_fscore_support(
            subset["gold_answerable"], subset["predicted_answerable"],
            labels=[0, 1], zero_division=0,
        )
        result = {
            "group": group, "n_examples": len(subset),
            "squad_v2_em": 100 * subset["exact_match"].mean(),
            "squad_v2_f1": 100 * subset["token_f1"].mean(),
            "answerability_accuracy": accuracy_score(subset["gold_answerable"], subset["predicted_answerable"]),
            "answerability_macro_f1": float(f1.mean()),
        }
        for label, name in enumerate(["unanswerable", "answerable"]):
            result.update({f"{name}_precision": precision[label], f"{name}_recall": recall[label],
                           f"{name}_f1": f1[label], f"n_{name}": int(support[label])})
        results.append(result)
    # QA EM/F1 use a 0-100 scale; answerability metrics use 0-1. Matching empty answers receive full QA credit.
    return pd.DataFrame(results), table

#  7. Main: joint training, validation selection, and saved outputs
def main():
# Run the experiment: prepare data, train, validate and select thresholds, then save the best model and results.
    # 1. Initialise the configuration, random seed, output directory, and device.
    config = CONFIG.copy()
    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "best_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Full XLM-R training is slow on CPU. A Colab GPU runtime is recommended.")

    # 2. [Added] Load the data and evaluate the always-empty baseline.
    data = load_project_data(config)
    print(f"Training examples: {len(data['train'])}; Validation examples: {len(data['validation'])}")
    empty_predictions = [{
        "id": ex["id"], "prediction_text": "", "prediction_start": -1,
        "prediction_end": -1, "predicted_answerable": 0,
    } for ex in data["validation"]]
    # This reference baseline requires no model, training, or threshold selection.
    empty_metrics, _ = evaluate_predictions(data["validation"], empty_predictions, config["languages"])
    empty_metrics.to_csv(output_dir / "empty_baseline_metrics.csv", index=False)

    # 3. Load the tokenizer and convert examples into window features.
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config["model_revision"], use_fast=True)
    features = {}
    for split in ["train", "validation"]:
        features[split] = data[split].map(
            prepare_features, batched=True, remove_columns=data[split].column_names,
            fn_kwargs={"tokenizer": tokenizer, "config": config, "training": split == "train"},
            desc=f"Preparing {split} windows",
        )
    train_features, val_features = features["train"], features["validation"]
    print(f"Training windows: {len(train_features)}; Validation windows: {len(val_features)}")

    # 4. [Added] Check answer coverage and the answer-length limit.
    # Check whether preprocessing or decoding limits prevent some gold answers from being fully represented or returned.
    # Count answerable examples with no fully covering training window, and covered answers exceeding the decoding token limit.
    covered_ids, long_ids = set(), set()
    for feature in train_features:
        if feature["start_positions"] != feature["cls_index"]:
            covered_ids.add(feature["example_id"])
            if feature["end_positions"] - feature["start_positions"] + 1 > config["max_answer_length"]:
                long_ids.add(feature["example_id"])
    answerable_ids = {ex["id"] for ex in data["train"] if ex["answerable"]}
    coverage = {
        "train_answerable_without_fully_containing_window": len(answerable_ids - covered_ids),
        "train_answerable_exceeding_decode_length_in_a_window": len(long_ids),
    }
    print("Training span coverage:", coverage)

    # 5. Create DataLoaders, load the model, and configure gradient checkpointing.
    train_loader = make_loader(train_features, config, training=True)
    val_loader = make_loader(val_features, config, training=False)
    model = AutoModelForQuestionAnswering.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    ).to(device)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # 6. [Lab 6 reused] Create the AdamW optimiser, scheduler, and scaler.
    # Follow Lab 6's AdamW parameter grouping:
    # Apply weight decay to ordinary weights, but not biases or LayerNorm weights.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if "bias" in name or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config["learning_rate"])
    updates_per_epoch = math.ceil(len(train_loader) / config["gradient_accumulation_steps"])
    total_updates = config["epochs"] * updates_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(config["warmup_ratio"] * total_updates),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"] and device.type == "cuda")

    # 7. [Added] Save configuration and runtime information for reproducibility.
    run_info = {
        "config": config, "python": platform.python_version(),
        "versions": {name: version(name) for name in ["torch", "transformers", "datasets", "numpy", "pandas", "scikit-learn"]},
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "dataset_fingerprints": {split: data[split]._fingerprint for split in data},
        "n_train_examples": len(data["train"]), "n_validation_examples": len(data["validation"]),
        "n_train_windows": len(train_features), "n_validation_windows": len(val_features),
        **coverage,
    }
    save_json(run_info, output_dir / "run_config.json")

    # 8. Train, validate, select a threshold, and evaluate after each epoch.
    # Each epoch follows this sequence:
    # 1. Train for one epoch, updating neural parameters and recording mean training loss.
    # 2. Predict start and end logits for each validation window.
    # 3. Aggregate windows to obtain each example's best span and no-answer score.
    # 4. Select the threshold maximising overall validation QA token F1.
    # 5. Apply the threshold to choose a text answer or an empty answer for each question.
    # 6. Compute overall and per-language QA and answerability metrics.
    # Reselect the threshold each epoch. Only step 1 updates neural parameters; later steps perform prediction, scoring, and selection.
    best_f1, best_epoch, history = -1.0, None, []
    total_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)
        training_seconds = time.perf_counter() - epoch_start
        logits = predict_logits(model, val_loader, device)
        candidates = decode_candidates(data["validation"], val_features, logits, config)
        threshold, curve = select_threshold(data["validation"], candidates)
        predictions = apply_threshold(candidates, threshold)
        metrics, table = evaluate_predictions(data["validation"], predictions, config["languages"])
        overall = metrics.iloc[0]
        history.append({
            "epoch": epoch, "train_loss": loss, "threshold": threshold,
            "validation_em": overall["squad_v2_em"], "validation_f1": overall["squad_v2_f1"],
            "validation_answerability_macro_f1": overall["answerability_macro_f1"],
            "training_seconds": training_seconds, "epoch_seconds": time.perf_counter() - epoch_start,
        })
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        print(f"Epoch {epoch:02d}/{config['epochs']} | Train loss: {loss:.4f} | "
              f"Val EM: {overall['squad_v2_em']:.2f} | Val F1: {overall['squad_v2_f1']:.2f} | "
              f"Answerability macro-F1: {overall['answerability_macro_f1']:.4f} | Threshold: {threshold:.4f}")

        # 9. [Added] Record epoch results and save the best epoch's model and threshold, rather than the final epoch's parameters.
        # Compare overall validation QA token F1; replace the checkpoint only on improvement, retaining the earlier epoch on ties.
        if overall["squad_v2_f1"] > best_f1 + 1e-10:
            best_f1, best_epoch = float(overall["squad_v2_f1"]), epoch
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            save_json({"threshold": threshold, "rule": "no_answer_score > threshold",
                       "objective": config["threshold_objective"], "best_epoch": epoch,
                       "config": config}, checkpoint_dir / "qa_settings.json")
            metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
            table.to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
            curve.to_csv(output_dir / "threshold_search.csv", index=False)

    # 10. Finalise the run record and print the best results.
    run_info.update({"best_epoch": best_epoch, "best_validation_f1": best_f1,
                     "training_and_validation_seconds": time.perf_counter() - total_start})
    save_json(run_info, output_dir / "run_config.json")
    print(f"\nBest epoch: {best_epoch}; validation SQuAD-v2 F1: {best_f1:.2f}")
    print(pd.read_csv(output_dir / "validation_metrics.csv").to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")

#  8. Reload the frozen model for new question/context pairs (no retraining needed)
def predict_new_examples(records, checkpoint_dir="w40_41_outputs/seed42/best_model"):
# Inference entry point: load the saved model and pass records containing the question/context pairs to predict.
# Records may contain newly constructed examples or existing validation examples.
    checkpoint_dir = Path(checkpoint_dir)
    settings = json.loads((checkpoint_dir / "qa_settings.json").read_text(encoding="utf-8"))
    config = settings["config"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    examples = Dataset.from_list(records)
    features = examples.map(prepare_features, batched=True, remove_columns=examples.column_names,
                            fn_kwargs={"tokenizer": tokenizer, "config": config, "training": False})
    logits = predict_logits(model, make_loader(features, config, training=False), device)
    candidates = decode_candidates(examples, features, logits, config)
    return pd.DataFrame(apply_threshold(candidates, settings["threshold"]))

if __name__ == "__main__":
    main()