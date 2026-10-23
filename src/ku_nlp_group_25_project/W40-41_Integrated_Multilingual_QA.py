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

# 首先定义CONFIG：集中保存各种设置的 Python 字典，其中包括和 Lab 相同的 base encoder，方便后续调用
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
    "max_question_tokens": 128,     # 极长问题截断，给 context 留出空间
    "stride": 128,                  # 相邻 context 窗口重叠的 token 数
    "n_best_size": 20,
    "max_answer_length": 30,        # 解码候选答案的最大 subword 数（沿用自Lab）
    "mixed_precision": True,
    "gradient_checkpointing": True, # 以额外计算换显存；不等于保存最佳 checkpoint
    "threshold_objective": "squad_v2_f1",  # 一个全局阈值，最大化整体 validation F1
    "max_train_examples": None,
    "max_validation_examples": None,
    "output_dir": "w40_41_outputs/seed44_epochs6",
}

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def set_seed(seed):
    # [Lab6 沿用] 固定初始化、shuffle 等随机过程；不同硬件不保证逐位相同。
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
        # example_id 是分语言后、抽样前的行号，保持与 W39 的顺序一致。再加上 split 前缀，防止 train/validation 的数字 ID 混淆。
        data = data.map(lambda row, idx: {
            "example_id": idx,
            "id": f"{split}-{idx}",
            # 原始 unanswerable 的 answer 可能是 yes/no，但本 task 中统一设为空。
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
    # [Lab6 沿用] 输入顺序是 (question, context)，其中sequence_ids==1 表示 context。
    # 使用预训练模型自己的 subword tokenizer；不再使用 W36/W38 的 split()（不 lowercase、不改 context，确保字符 offsets 始终对应原始文本。）
    questions = []
    for question in examples["question"]:
        q_offsets = tokenizer(question, add_special_tokens=False,
                              return_offsets_mapping=True)["offset_mapping"]
        if len(q_offsets) > config["max_question_tokens"]:
            # 截断超长问题
            question = question[:q_offsets[config["max_question_tokens"] - 1][1]]
        questions.append(question)

    encoded = tokenizer(
        questions, examples["context"],
        max_length=config["max_length"], truncation="only_second",
        stride=config["stride"], padding="max_length",
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    # 逐个处理切分后的窗口，为训练准备答案的起始和结束token的位置
    # 具体来说：答案的字符位置 → start_position 和 end_position 两个标签
    sample_map = encoded.pop("overflow_to_sample_mapping")
    example_ids, cls_indices = [], []
    starts, ends = [], []

    # i：当前窗口在分词结果中的编号；sample_index：当前窗口对应的原始样本在本批数据中的编号
    for i, sample_index in enumerate(sample_map):
        example_ids.append(examples["id"][sample_index])
        # 找到 CLS 特殊 token 在当前输入中的位置。后面用这个位置代表“当前窗口没有答案”。
        cls_index = encoded["input_ids"][i].index(tokenizer.cls_token_id)
        cls_indices.append(cls_index)
        # 取：每个 token 属于 question（0）、context（1），还是特殊 token / padding（None）。
        sequence_ids = encoded.sequence_ids(i)
        # 取：每个 token 对应的原文字符区间；对于 context token，它对应原始 context 的字符位置。
        offsets = encoded["offset_mapping"][i]
        # 取：context tokens 在当前输入中的索引（token index），方便后面只在 context 范围内定位答案。
        context_indices = []
        for j, sid in enumerate(sequence_ids):
            if sid == 1:
                context_indices.append(j)

        if training:
            # 默认当前窗口没有答案：将起、终点设为 CLS/CLS，再检查是否需要改成实际答案的位置。
            # 两种情况下继续保持CLS：
            # 1) 原样本就 unanswerable；2) 原样本answerable，但当前窗口没包含完整答案。
            # 2)是窗口级标签，同一个 answerable 样本产生的不同窗口，可以有不同的训练标签。
            start_position = end_position = cls_index
            answers = examples["answers"][sample_index]
            if answers["text"] and context_indices:
                # 如果：原样本有答案，且当前窗口有context tokens
                # 则：先取答案在原始 context 中的字符区间，再取当前窗口第一个和最后一个 context token
                # 检查：窗口开始位置不晚于答案开头，窗口结束位置不早于答案结尾。
                # 将答案的字符位置换成 token 位置，通过两个 while，分别从左向右找答案起始token，从右向左找答案结束token
                # 注意区分：token标签的 end_position 和字符区间的 char_end 不同。
                # 最后，保存两个训练标签。每个窗口最终都有一对标签：实际答案的起止 token，或者表示无答案的 CLS / CLS。
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

        # encode筛选：只选择 context，question、special tokens、padding 均屏蔽。
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
    # [Lab6 沿用] 删除本 batch 用不到的右侧 padding，以减少计算量。
    width = max(sum(row["attention_mask"]) for row in rows)
    batch = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in ["input_ids", "attention_mask", "token_type_ids"]:
            values = [value[:width] for value in values]
        batch[key] = torch.tensor(values, dtype=torch.long)
    return batch

def make_loader(features, config, training):
# 把已经处理好的窗口数据，组织成模型可以逐批读取的 DataLoader
# 具体来说，决定：给模型哪些列、每批多少条、是否打乱顺序，以及怎样组装每一批。
    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in features.column_names:
        columns.append("token_type_ids")
    if training:
        # 训练时传入正确答案的起止位置，用于计算 loss
        columns += ["start_positions", "end_positions"]
    # metadata 不传入 model；validation 不含 gold start/end，避免泄漏。
    return DataLoader(
        # 仅保留上面选中的列，其他信息仍保存在原来的 features 中，供后处理使用，但不交给模型。
        features.select_columns(columns),
        # 设置每批数据的数量，区分训练/验证
        batch_size=config["train_batch_size" if training else "eval_batch_size"],
        # 训练时打乱顺序，用前面定义的方法组装一个batch（PyTorch tensors）
        shuffle=training, collate_fn=collate_batch,
    )

#  4. Training and inference
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, config):
    # [Lab6 沿用] 把训练集中的所有窗口按 batch 遍历一遍，在这个过程中计算 loss、反向传播并更新模型参数，最后返回这一轮的平均训练 loss
    # optimizer 负责根据梯度更新参数，learning rate 控制更新的步幅。
    # scheduler 则负责随着训练进行，改变 optimizer 使用的 learning rate。
    # scaler 帮助 FP16 训练保持数值稳定
    model.train()
    # 清除上一轮计算的梯度
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_features = 0.0, 0
    # 规定累计几个batch的梯度后，更新一次参数
    accumulation = config["gradient_accumulation_steps"]
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        for key, value in batch.items():
            batch[key] = value.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            # 前向计算，得到预测与loss
            loss = model(**batch).loss
        batch_size = batch["input_ids"].shape[0]
        # 记录当前 batch 的 loss (batch的平均loss*batch的窗口数)，再累加
        total_loss += loss.item() * batch_size
        total_features += batch_size
        # [新增] 反向传播，先梯度累积
        # 比如共有5个batch，第一、二累计组处理两个（group_size=2），最后一组只有一个batch，不再除2
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        scaler.scale(loss / group_size).backward()
        # 满足条件（已累计规定数量的 batch/已经到达本轮最后一个 batch）才更新参数
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # FP16 溢出时会跳过 optimizer 更新，此时也不推进 scheduler。
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            # 当参数更新没有因溢出而跳过，才推进学习率计划。scheduler 控制的是后续更新使用的学习率。
            optimizer.zero_grad(set_to_none=True)
    # 返回本轮各个 batch 在训练过程中测得的平均 loss。模型参数已经在函数内部被更新，main() 接下来会拿更新后的模型进行验证。
    return total_loss / total_features

@torch.no_grad()
def predict_logits(model, loader, device):
    # [Lab6 沿用] 评估模式做预测时，为了区别于训练阶段，进行以下调整：
    # 训练时，dropout会随机将一部分中间表示置为 0，帮助减少过拟合；验证／预测时，不需要这种随机丢弃
    # 训练时，PyTorch 需要记录前向计算中各个运算之间的关系，以及计算梯度所需的信息。这些构成了用于自动求导的计算图。
    # 预测时，不需要反向传播，因此用no_grad表示：这个函数运行时，不跟踪用于自动求导的计算过程
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
    # [Lab6 沿用] 将同一原始样本的所有滑窗聚合，搜索合法答案 span。
    # 具体步骤：收集窗口 → 检查每个窗口 → 筛去不合法的答案 → 选出最佳文本答案 → 计算无答案倾向
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
            # [新增] 除了寻找“最可能的答案片段”，还计算“没有答案”的分数，让后续能够在两者之间做选择。
            # null score = 起点在 CLS 的分数+终点在 CLS 的分数
            # span score = 答案起点分数+答案终点分数
            # HF 常用聚合：跨窗口取最小 null score（最不支持“无答案”的那个窗口的分数），span 取最大 score（最有竞争力的答案）。
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
                        # 直接切原始 context，保留空格/标点；不用 tokenizer.decode 拼答案。
                        best_text = example["context"][best_start:best_end]
        candidates.append({
            "id": example["id"], "best_span_text": best_text,
            "best_span_start": best_start, "best_span_end": best_end,
            "best_span_score": best_score, "null_score": null_score,
            # 正值越大，越偏向 no answer，越倾向于无答案。这是 logit score 差，不是校准概率。
            "no_answer_score": null_score - best_score,
        })
    # 仅保存最佳片段和上述分数，后续再与 validation 上选定的阈值比较，决定返回空答案/最佳答案片段
    return candidates

#  6. SQuAD-v2 scoring and threshold selection
# 给答案打分 → 在 validation 上选出“是否回答”的阈值 → 生成最终预测 → 汇总评估结果。（模型参数在这个过程中不会更新）
# [Lab6 修改] 前三个函数延续了 Lab 6 的答案评估思路，并补上空答案的处理；后面的函数用于完成项目需要的阈值选择和综合评估。
def normalise_answer(text):
# 统一英文答案的比较格式
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def score_text(prediction, gold):
# 比较一个预测答案和一个正确答案，返回两个分数
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    exact = float(pred_tokens == gold_tokens)
    if not pred_tokens or not gold_tokens:
        return exact, float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    f1 = 2.0 * overlap / (len(pred_tokens) + len(gold_tokens))
    return exact, f1

def score_against_references(prediction, answers):
# 处理一个问题对应多个正确答案的情况
    references = [text for text in answers["text"] if normalise_answer(text)] or [""]
    scores = [score_text(prediction, gold) for gold in references]
    return max(score[0] for score in scores), max(score[1] for score in scores)

def select_threshold(examples, candidates):
# [新增] 在 validation 上选择回答阈值：只用 validation gold 找一个全局阈值，让平均 QA token F1最高。
    # 具体机制
    # 第一步：预先算出每道题的两种得分，empty_f1和span_f1。
    # 第二步：从“所有问题都不回答”开始。初始阈值设置得比所有有限的Δ都小。
    # 第三步：逐渐提高阈值，让问题依次从“不回答”切换为“回答”。
    # 第四步：保留平均 F1 最高的阈值。
    deltas = np.array([row["no_answer_score"] for row in candidates], dtype=float)
    empty_f1 = np.array([score_against_references("", ex["answers"])[1] for ex in examples])
    span_f1 = np.array([score_against_references(row["best_span_text"], ex["answers"])[1]
                        for ex, row in zip(examples, candidates)])
    finite = deltas[np.isfinite(deltas)]
    # 没有合法候选的样本始终为空；不会因提高阈值而被强制输出乱码。
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
# 将选好的阈值应用到预测上，真正生成最终答案
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
#  [新增] 生成逐条预测表和整体评估表
    # 同时完成两个层次的评估：答案内容是否正确，以及是否应该回答的判断是否正确。
    # 返回：汇总指标表：用于报告整体和各语言表现；逐条预测明细表：用于后续人工 error analysis。
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
        if len(subset) == 0:  # 只可能在小样本 smoke run 时缺少某语言。
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
    # 输出尺度：QA的EM/F1 是 0–100；answerability 指标是 0–1。空答案匹配给 QA 满分。
    return pd.DataFrame(results), table

#  7. Main: joint training, validation selection, and saved outputs
def main():
# 完成一次完整实验：准备数据 → 训练 → 验证和选择阈值 → 保存最佳模型与结果。
    # 1 初始化配置、随机种子、保存路径和计算设备
    config = CONFIG.copy()
    set_seed(config["seed"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "best_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Full XLM-R training is slow on CPU. A Colab GPU runtime is recommended.")

    # 2 [新增] 加载数据，并计算“始终返回空答案”的 baseline
    data = load_project_data(config)
    print(f"Training examples: {len(data['train'])}; Validation examples: {len(data['validation'])}")
    empty_predictions = [{
        "id": ex["id"], "prediction_text": "", "prediction_start": -1,
        "prediction_end": -1, "predicted_answerable": 0,
    } for ex in data["validation"]]
    # 这个 baseline 不需要模型、训练或阈值选择。它只提供一个简单参照
    empty_metrics, _ = evaluate_predictions(data["validation"], empty_predictions, config["languages"])
    empty_metrics.to_csv(output_dir / "empty_baseline_metrics.csv", index=False)

    # 3 加载 tokenizer，将原始样本转换为窗口特征
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

    # 4 [新增] 检查窗口和答案长度限制的影响
    # 具体来说：当前预处理和答案搜索设置，会不会让部分正确的答案无法被完整学习或输出。
    # 统计：1 原本有答案，但没有任何训练窗口完整包含答案的样本数。 2 答案虽然被窗口完整覆盖，但 token 数超过搜索允许的最大答案长度。
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

    # 5 创建 DataLoader，加载模型，设置 gradient checkpointing
    train_loader = make_loader(train_features, config, training=True)
    val_loader = make_loader(val_features, config, training=False)
    model = AutoModelForQuestionAnswering.from_pretrained(
        config["model_name"], revision=config["model_revision"]
    ).to(device)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # 6 [Lab6 沿用] 创建 AdamW optimizer、scheduler 和 scaler
    # 对于AdamW，延续 Lab 6 的参数分组方式：
    # 一般权重使用设置的 weight decay；bias 和 LayerNorm 的权重不使用 weight decay。
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

    # 7 [新增] 保存本次实验的运行信息，作为这次实验的记录
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

    # 8 每一轮执行训练、验证、阈值选择和评估
    # 每个 epoch 按下面的顺序进行：
    # 1 训练一轮：使用 training 数据更新模型参数，得到平均训练 loss。
    # 2 验证集预测：输出各窗口中每个位置的起点、终点分数。
    # 3 答案搜索：汇总同一样本的窗口，得到最佳片段和无答案分数。
    # 4 选择阈值：在 validation 上寻找整体 QA Token F1 最高的阈值。
    # 5 应用阈值：为每个问题确定返回文本答案还是空答案。
    # 6 评估结果：计算整体和各语言的 QA 与 answerability 指标。
    # 注意：每个 epoch 都会重新选择阈值。只有第一步更新神经网络参数；后面的步骤用于预测、评分和选择。
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

        # 9 [新增] 记录每轮表现，保存最佳 epoch 的模型和当时的阈值，而不是最后一轮参数。
        # 用来对比的是每轮的的整体 validation QA Token F1，每次对比保存更高者，相同时保留较早 epoch。
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

    # 10 训练结束后更新记录，打印最佳结果
    run_info.update({"best_epoch": best_epoch, "best_validation_f1": best_f1,
                     "training_and_validation_seconds": time.perf_counter() - total_start})
    save_json(run_info, output_dir / "run_config.json")
    print(f"\nBest epoch: {best_epoch}; validation SQuAD-v2 F1: {best_f1:.2f}")
    print(pd.read_csv(output_dir / "validation_metrics.csv").to_string(index=False))
    print(f"\nOutputs saved to: {output_dir.resolve()}")

#  8. Reload the frozen model for new question/context pairs (no retraining needed)
def predict_new_examples(records, checkpoint_dir="w40_41_outputs/seed42/best_model"):
# 以后加载已保存模型进行预测的入口，records 是预测新问题时，需要提供的question/context pairs数据
# 预测对象可以是新的自己定义的数据，也可以是已有 validation 中的样本
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