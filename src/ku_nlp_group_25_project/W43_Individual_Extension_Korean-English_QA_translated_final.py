# W43: Translated Korean questions; shared implementation comments are in the W40-41 QA script.
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


CONFIG = {
    "model_name": "FacebookAI/xlm-roberta-base",
    "model_revision": "main",
    "data_path": r"D:\Python\nlp_data\data\tydi_xor_rc_ar_ko_te\ko_questions_english",
    "languages": ["ko"],
    "seed": 44,
    "epochs": 6,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_length": 384,              
    "max_question_tokens": 128,     
    "stride": 128,                  
    "n_best_size": 20,
    "max_answer_length": 30,        
    "mixed_precision": True,
    "gradient_checkpointing": True, 
    "threshold_objective": "squad_v2_f1",  
    "max_train_examples": None,
    "max_validation_examples": None,
    "output_dir": "w43_t_outputs/seed44_epochs6",
}

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_project_data(config):
    data_path = Path(config["data_path"])
    raw = DatasetDict({
        "train": Dataset.from_file(str(data_path / "train.arrow")),
        "validation": Dataset.from_file(str(data_path / "validation.arrow")),
    })
    result = {}
    for split in ["train", "validation"]:
        data = raw[split].filter(lambda row: row["lang"] in config["languages"])
        
        data = data.map(lambda row, idx: {
            "example_id": idx,
            "id": f"{split}-{idx}",
            
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



def prepare_features(examples, tokenizer, config, training):
    
    
    questions = []
    for question in examples["question"]:
        q_offsets = tokenizer(question, add_special_tokens=False,
                              return_offsets_mapping=True)["offset_mapping"]
        if len(q_offsets) > config["max_question_tokens"]:
            
            question = question[:q_offsets[config["max_question_tokens"] - 1][1]]
        questions.append(question)

    encoded = tokenizer(
        questions, examples["context"],
        max_length=config["max_length"], truncation="only_second",
        stride=config["stride"], padding="max_length",
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    
    
    sample_map = encoded.pop("overflow_to_sample_mapping")
    example_ids, cls_indices = [], []
    starts, ends = [], []

    
    for i, sample_index in enumerate(sample_map):
        example_ids.append(examples["id"][sample_index])
        
        cls_index = encoded["input_ids"][i].index(tokenizer.cls_token_id)
        cls_indices.append(cls_index)
        
        sequence_ids = encoded.sequence_ids(i)
        
        offsets = encoded["offset_mapping"][i]
        
        context_indices = []
        for j, sid in enumerate(sequence_ids):
            if sid == 1:
                context_indices.append(j)

        if training:
            
            
            
            
            start_position = end_position = cls_index
            answers = examples["answers"][sample_index]
            if answers["text"] and context_indices:
                
                
                
                
                
                
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
    
    width = max(sum(row["attention_mask"]) for row in rows)
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in ["input_ids", "attention_mask", "token_type_ids"]:
            values = [value[:width] for value in values]
        batch[key] = torch.tensor(values, dtype=torch.long)
    return batch

def make_loader(features, config, training):


    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in features.column_names:
        columns.append("token_type_ids")
    if training:
        
        columns += ["start_positions", "end_positions"]
    
    return DataLoader(
        
        features.select_columns(columns),
        
        batch_size=config["train_batch_size" if training else "eval_batch_size"],
        
        shuffle=training, collate_fn=collate_batch,
    )


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, config):
    
    
    
    
    model.train()
    
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_features = 0.0, 0
    
    accumulation = config["gradient_accumulation_steps"]
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        for key, value in batch.items():
            batch[key] = value.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            
            loss = model(**batch).loss
        batch_size = batch["input_ids"].shape[0]
        
        total_loss += loss.item() * batch_size
        total_features += batch_size
        
        
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        scaler.scale(loss / group_size).backward()
        
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
    
    return total_loss / total_features

@torch.no_grad()
def predict_logits(model, loader, device):
    
    
    
    
    model.eval()
    start_logits, end_logits = [], []
    for batch in tqdm(loader, desc="Predicting"):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        start_logits.extend(outputs.start_logits.float().cpu().numpy())
        end_logits.extend(outputs.end_logits.float().cpu().numpy())
    return start_logits, end_logits



def decode_candidates(examples, features, logits, config):
    
    
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
                        
                        best_text = example["context"][best_start:best_end]
        candidates.append({
            "id": example["id"], "best_span_text": best_text,
            "best_span_start": best_start, "best_span_end": best_end,
            "best_span_score": best_score, "null_score": null_score,
            
            "no_answer_score": null_score - best_score,
        })
    
    return candidates




def normalise_answer(text):

    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def score_text(prediction, gold):

    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    exact = float(pred_tokens == gold_tokens)
    if not pred_tokens or not gold_tokens:
        return exact, float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    f1 = 2.0 * overlap / (len(pred_tokens) + len(gold_tokens))
    return exact, f1

def score_against_references(prediction, answers):

    references = [text for text in answers["text"] if normalise_answer(text)] or [""]
    scores = [score_text(prediction, gold) for gold in references]
    return max(score[0] for score in scores), max(score[1] for score in scores)

def select_threshold(examples, candidates):

    
    
    
    
    
    deltas = np.array([row["no_answer_score"] for row in candidates], dtype=float)
    empty_f1 = np.array([score_against_references("", ex["answers"])[1] for ex in examples])
    span_f1 = np.array([score_against_references(row["best_span_text"], ex["answers"])[1]
                        for ex, row in zip(examples, candidates)])
    finite = deltas[np.isfinite(deltas)]
    
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

    predictions = []
    for row in candidates:
        no_answer = row["no_answer_score"] > threshold or row["best_span_text"] == ""
        predictions.append({
            **row,
            "prediction_text": "" if no_answer else row["best_span_text"],
            "prediction_start": -1 if no_answer else row["best_span_start"],
            "prediction_end": -1 if no_answer else row["best_span_end"],  
            "predicted_answerable": int(not no_answer),
        })
    return predictions

def evaluate_predictions(examples, predictions, languages):

    
    
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
        if len(subset) == 0:  
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
    
    return pd.DataFrame(results), table


def main():

    
    config = CONFIG.copy()
    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "best_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Full XLM-R training is slow on CPU. A Colab GPU runtime is recommended.")

    
    data = load_project_data(config)
    print(f"Training examples: {len(data['train'])}; Validation examples: {len(data['validation'])}")
    empty_predictions = [{
        "id": ex["id"], "prediction_text": "", "prediction_start": -1,
        "prediction_end": -1, "predicted_answerable": 0,
    } for ex in data["validation"]]
    
    empty_metrics, _ = evaluate_predictions(data["validation"], empty_predictions, config["languages"])
    empty_metrics.to_csv(output_dir / "empty_baseline_metrics.csv", index=False)

    
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

    
    train_loader = make_loader(train_features, config, training=True)
    val_loader = make_loader(val_features, config, training=False)
    model = AutoModelForQuestionAnswering.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    ).to(device)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    
    
    
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

    
    run_info.update({"best_epoch": best_epoch, "best_validation_f1": best_f1,
                     "training_and_validation_seconds": time.perf_counter() - total_start})
    save_json(run_info, output_dir / "run_config.json")
    print(f"\nBest epoch: {best_epoch}; validation SQuAD-v2 F1: {best_f1:.2f}")
    print(pd.read_csv(output_dir / "validation_metrics.csv").to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")


def predict_new_examples(records, checkpoint_dir="w40_41_outputs/seed42/best_model"):


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