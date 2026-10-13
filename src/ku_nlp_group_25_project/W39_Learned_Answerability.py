from collections import Counter
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np
import pandas as pd

from datasets import load_from_disk
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# 1. Shared data preparation

DATA_PATH = r"D:\Python\nlp_data\data\tydi_xor_rc_ar_ko_te"
LANGUAGES = ["ar", "ko", "te"]

NUMERIC_FEATURE_NAMES = [
    "question_length",
    "context_length",
    "shared_token_count",
    "question_overlap_ratio",
]

def load_data():
    dataset = load_from_disk(DATA_PATH)

    columns = ["lang", "question", "context", "answerable"]

    train = dataset["train"].to_pandas()[columns].copy()
    validation = dataset["validation"].to_pandas()[columns].copy()

    # 保留样本编号，方便之后定位错误样本。
    train["example_id"] = train.index
    validation["example_id"] = validation.index

    return train, validation

# 2. Model 1: feature extraction

def tokenize(text):
    return text.lower().split()

def extract_numeric_features(frame):
    rows = []

    for _, row in frame.iterrows():
        question_tokens = tokenize(row["question"])
        context_tokens = tokenize(row["context"])

        question_length = len(question_tokens)
        context_length = len(context_tokens)

        question_set = set(question_tokens)
        context_set = set(context_tokens)

        shared_token_count = len(question_set & context_set)

        question_overlap_ratio = (shared_token_count / len(question_set))

        rows.append([
            question_length,
            context_length,
            shared_token_count,
            question_overlap_ratio,
        ])

    return np.array(rows, dtype=float)

def build_model1_features(train, validation):
    # question 和 context 分别建立词表、计算 TF-IDF。
    # min_df=2：词至少出现在两篇 training 文档中才保留。

    question_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        min_df=2,
    )

    context_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        min_df=2,
    )

    # 只在 training data 上学习词表（fit）和 IDF（transform）。
    # 例：training 有 6,335 条 question，训练后保留了1,000个词，结果形状为：(6335, 1000)
    # 表示：每行是一条question，每列对应词表中一个token，单元格value是该token在这条 question 中的 TF-IDF 权重
    train_question_features = question_vectorizer.fit_transform(
        train["question"]
    )
    train_context_features = context_vectorizer.fit_transform(
        train["context"]
    )

    # validation 使用 training 已经学到的规则，因此仅transform，不用再次fit
    validation_question_features = question_vectorizer.transform(
        validation["question"]
    )
    validation_context_features = context_vectorizer.transform(
        validation["context"]
    )

    # 得到 numeric features
    train_numeric = extract_numeric_features(train)
    validation_numeric = extract_numeric_features(validation)

    # 不同numeric features的尺度可能差别很大，比如context_length可能是几百，而ratio在0-1，因此需要标准化数值特征。
    # 这是训练 Logistic Regression 时常见的预处理，主要有两个作用：1. 有利于优化算法收敛； 2. 让正则化更合理
    # 和TF-IDF一样，均值、标准差也只从 training 学习。
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train_numeric)
    validation_numeric = scaler.transform(validation_numeric)

    # 把三列数据拼成一列：[question TF-IDF | context TF-IDF | numeric features]
    # TF-IDF 中有大量 0，已经是稀疏矩阵。把train_numeric从普通Numpy数据转为稀疏矩阵，相同格式方便拼接。
    X_train = hstack([
        train_question_features,
        train_context_features,
        csr_matrix(train_numeric),
    ], format="csr")

    X_validation = hstack([
        validation_question_features,
        validation_context_features,
        csr_matrix(validation_numeric),
    ], format="csr")

    # 特征名顺序与拼接后的列保持顺序一致，方便之后解释每个系数对应的特征
    feature_names = []

    for token in question_vectorizer.get_feature_names_out():
        feature_names.append(f"question::{token}")

    for token in context_vectorizer.get_feature_names_out():
        feature_names.append(f"context::{token}")

    feature_names.extend(NUMERIC_FEATURE_NAMES)

    # 保存已拟合的预处理器，之后对新数据预测时可直接调用。
    preprocessors = {
        "question_vectorizer": question_vectorizer,
        "context_vectorizer": context_vectorizer,
        "scaler": scaler,
    }

    return X_train, X_validation, feature_names, preprocessors

# 3. Model 1: training

def train_model1(X_train, y_train):
    model = LogisticRegression(
    # C 是正则化强度的倒数，之后可通过validation 比较不同设置
    # class_weight="balanced"：training 中 unanswerable 较少，因此增加少数类样本的权重。也可以在 validation数据上对比 =None
    # solver 指用什么优化算法求解模型的系数和截距。
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(X_train, y_train)
    return model

# 4. Shared evaluation for both models (！！！)
# 包含：整体评估、分语言评估和预测表整理函数

# 准备对模型预测评价指标
def print_metrics(y_true, y_pred, title):
    print(f"\n===== {title} =====")

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    print(f"Macro-F1: {macro_f1:.4f}")

    # 0 = unanswerable，1 = answerable。
    # 输出两个类别分别对应的 precision / recall / F1。
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["unanswerable", "answerable"],
        digits=4,
        zero_division=0,
    ))

# 比较真实标签和模型预测标签：先评估完整validation，再分三种语言单独评估。
def evaluate_classifier(validation, predictions, model_name):
    y_true = validation["answerable"].astype(int).to_numpy()
    predictions = np.asarray(predictions)

    print_metrics(
        y_true,
        predictions,
        f"{model_name}: Overall Validation",
    )

    for lang in LANGUAGES:
        mask = (validation["lang"] == lang).to_numpy()

        print_metrics(
            y_true[mask],
            predictions[mask],
            f"{model_name}: Validation ({lang})",
        )

# 关键函数！！！整理模型预测结果，并给结果结果加各种标签，用于后续错误分析。
def make_prediction_table(validation, predictions, probabilities):
    # probabilities 是模型给出的 P(answerable)。
    results = validation[
        ["example_id", "lang", "question", "context"]
    ].copy()

    results["gold"] = (
        validation["answerable"].astype(int).to_numpy()
    )
    results["prediction"] = predictions
    results["p_answerable"] = probabilities

    # confidence 表示模型给“预测类别”的概率，不是经过额外校准的正确率。
    # np.where(条件, 条件成立时的值, 条件不成立时的值)：如果预测为answerable，使用P(answerable)，否则使用P(unanswerable)
    # 例：P(answerable) = 0.90 预测类别：1（answerable） confidence：0.90
    #     P(answerable) = 0.20 预测类别：0（unanswerable） confidence：1-0.20 = 0.80
    results["confidence"] = np.where(
        predictions == 1,
        probabilities,
        1 - probabilities,
    )

    results["correct"] = (
        results["gold"] == results["prediction"]
    )

    return results

# 5. Model 1: coefficient inspection

def inspect_coefficients(model, feature_names, top_n=10):
    coefficients = pd.DataFrame({
        "feature": feature_names,
        # P(answerable)=sigma(b+w_1x_1+...+w_dx_d)
        # model_coef_：取出模型学到的特征系数，也就是这些 w，一个形状为 (1, 特征数量) 的二维数组，[0] 取出其中唯一的一行。
        "coefficient": model.coef_[0],
    })

    # 分别取出系数最大、最小的几个特征。正系数推动预测 answerable；负系数推动预测 unanswerable。
    print("\n===== Features Toward Answerable =====")
    print(
        coefficients.nlargest(top_n, "coefficient")
        .to_string(index=False)
    )

    print("\n===== Features Toward Unanswerable =====")
    print(
        coefficients.nsmallest(top_n, "coefficient")
        .to_string(index=False)
    )

    # 最大、最小系数，是从全部三类特征里选出的。
    # 额外筛出4个之前手动构造的数值特征（两个长度、共有 token 数、重合比例），确保即使它们即使没进入前10，也打印出来方便检查。
    print("\n===== Numeric Feature Coefficients =====")
    print(
        coefficients[
            coefficients["feature"].isin(NUMERIC_FEATURE_NAMES)
        ].to_string(index=False)
    )

    return coefficients

# 6. Model 2: settings and reproducibility

MODEL2_CONFIG = {
    "seed": 42,
    "min_freq": 2,
    "embedding_dim": 64,
    "hidden_dim": 64,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
}
# 0和1只是编号，不代表数值大小或重要程度，普通token从2开始
PAD_ID = 0 # <PAD> 的编号是 0，用于补齐同一 batch 中长短不一的序列
UNK_ID = 1 # <UNK> 的编号是 1，代表不在词表中的 token
# 创建一个用于接收结果的文件夹
# 保存：model1_predictions.csv、model2_predictions.csv、model2_training_history.csv、model2_bilstm.pt、model_comparison.csv
OUTPUT_DIR = Path(__file__).resolve().parent / "w39_outputs"


def set_seed(seed):
    # Model2的随机性共有三处：
    # 1 参数初始化：embedding、LSTM 和分类层的参数一开始随机生成。
    # 2 训练数据顺序：shuffle=True 会打乱每轮训练的样本顺序。
    # 3 Dropout：训练时随机将部分激活值置零。
    # 固定seed的目的是：尽量在重复运行时，保证这些随机过程按相同的顺序发生
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 要求 cuDNN 使用确定性的算法，帮助相同输入和设置得到可重复的结果。
    torch.backends.cudnn.deterministic = True
    # 关闭运行时的算法测速选优，避免因为测速波动而在不同运行中选择不同算法。
    torch.backends.cudnn.benchmark = False

# 7. Model 2: vocabulary, Dataset and batches
def build_vocabulary(train, min_freq):
    # 只统计 training 的 question 和 context。
    # 两种文本共享一个词表；复用 Model 1 的 lower().split()。
    # min_freq 指 token 总出现次数。
    counts = Counter()
    for column in ["question", "context"]:
        for text in train[column]:
            counts.update(tokenize(text))

    vocabulary = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for token, count in counts.items():
        if count >= min_freq:
            vocabulary[token] = len(vocabulary)
    return vocabulary


class AnswerabilityDataset(Dataset):
    # 定义一个类，对应 ClassificationDatasetReader 的结构
    # 可以理解为：一个存放处理后样本的容器，并提供“有多少条”和“取第几条”的方法，供之后的 DataLoader 使用。
    # 把 pandas 表格中的每条样本，转换为 PyTorch 后续训练可以读取的格式。
    # 每行数据最终变成：(question_token_ids, context_token_ids, answerable_label)
    def __init__(self, frame, vocabulary):
    # 处理并保存全部样本
        self.examples = []
        for _, row in frame.iterrows():
            question_ids = []
            for token in tokenize(row["question"]):
                question_ids.append(vocabulary.get(token, UNK_ID))

            context_ids = []
            for token in tokenize(row["context"]):
                context_ids.append(vocabulary.get(token, UNK_ID))

            self.examples.append((
                torch.tensor(question_ids, dtype=torch.long),
                torch.tensor(context_ids, dtype=torch.long),
                int(row["answerable"]),
            ))

    def __len__(self):
    # 告诉程序有多少条样本
        return len(self.examples)

    def __getitem__(self, index):
    # 取出指定位置的样本
        return self.examples[index]


def collate_model2_batch(examples):
    # [Lab 4 adapted] 对应 collate_batch_transformer 的 batch padding。
    questions = []
    contexts = []
    labels = []
    question_lengths = []
    context_lengths = []
    for question, context, label in examples:
        questions.append(question)
        contexts.append(context)
        labels.append(label)
        question_lengths.append(len(question))
        context_lengths.append(len(context))

    return (
        pad_sequence(questions, batch_first=True, padding_value=PAD_ID),
        pad_sequence(contexts, batch_first=True, padding_value=PAD_ID),
        torch.tensor(question_lengths, dtype=torch.long),
        torch.tensor(context_lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

# 8. Model 2: recurrent neural classifier
class RecurrentAnswerabilityClassifier(nn.Module):
    # 建立RNN模型：从头训练的 embedding + 两个独立的 BiLSTM。
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        # 先用super()建立模型的基础管理结构，在用self分别登记 embedding 层、 question 编码器、 context 编码器和分类网络
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=PAD_ID
        )
        self.question_lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.context_lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        # 每个双向 LSTM 的表示长度为 2 * hidden_dim，两段文本拼接后为 4 * hidden_dim。
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, token_ids, lengths, lstm):
        # 把一段文本变成一个向量
        # token_ids: [batch_size, sequence_length]
        # embedded:  [batch_size, sequence_length, embedding_dim]
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = lstm(packed)
        # 单层双向 LSTM: hidden[0] 为正向最终状态，hidden[1] 为反向。
        return torch.cat([hidden[0], hidden[1]], dim=1)

    def forward(self, question_ids, context_ids, question_lengths, context_lengths):
        # 通过调用两次encode，编码问题和上下文，组合整个模型的计算流程
        question_vector = self.encode(
            question_ids, question_lengths, self.question_lstm
        )
        context_vector = self.encode(
            context_ids, context_lengths, self.context_lstm
        )
        combined = torch.cat([question_vector, context_vector], dim=1)
        # 输出两个未归一化分数 logits，对应 [unanswerable, answerable]。
        # CrossEntropyLoss 内部处理 log-softmax，训练前不用手动 softmax。
        return self.classifier(combined)


# 9. Model 2: validation and training
def predict_model2(model, loader, device, criterion):
    #
    model.eval()
    probabilities = []
    total_loss = 0.0
    with torch.no_grad():
        for questions, contexts, q_lengths, c_lengths, labels in loader:
            logits = model(
                questions.to(device), contexts.to(device), q_lengths, c_lengths
            )
            losses = criterion(logits, labels.to(device))
            total_loss += losses.sum().item()
            # 第 1 列对应 answerable，与 Model 1 的概率列一致。
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(batch_probabilities.cpu().tolist())
    return np.asarray(probabilities), total_loss / len(loader.dataset)


def train_model2(train, validation, config):
    # 1. 准备数据和模型
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = build_vocabulary(train, config["min_freq"])
    train_dataset = AnswerabilityDataset(train, vocabulary)
    validation_dataset = AnswerabilityDataset(validation, vocabulary)

    # training 打乱顺序，validation 不打乱，因为validation 顺序必须与原 DataFrame 一致，才可复用 prediction table。
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_model2_batch, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_model2_batch, num_workers=0,
    )
    model = RecurrentAnswerabilityClassifier(
        len(vocabulary), config["embedding_dim"],
        config["hidden_dim"], config["dropout"],
    ).to(device)

    y_train = train["answerable"].astype(int).to_numpy()
    class_counts = np.bincount(y_train, minlength=2)
    weights = len(y_train) / (2.0 * class_counts)
    # 2. 设置如何”学习“
    # criterion：损失函数，衡量预测与真实标签的差距。
    # optimizer：优化器，根据梯度更新模型参数
    # 然后准备记录最佳分数、最佳轮次、最佳参数，以及每轮训练结果的变量。
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        reduction="none",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    y_validation = validation["answerable"].astype(int).to_numpy()
    best_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    history = []

    print("\n===== Model 2: BiLSTM Training =====")
    print(f"Device: {device}")
    print(f"Vocabulary size (including PAD/UNK): {len(vocabulary)}")
    print(f"Class weights [unanswerable, answerable]: {weights}")
    print(f"Configuration: {config}")

    # 3. 真正进行训练的两层循环
    # 外层：训练多少轮  内层：这一轮里，逐个batch处理，遍历整个training set
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        for questions, contexts, q_lengths, c_lengths, labels in train_loader:
            # 清除上一次的梯度，避免意外累加
            optimizer.zero_grad()
            # 前向计算，得到两个类别分数
            logits = model(
                questions.to(device), contexts.to(device), q_lengths, c_lengths
            )
            # 计算每条样本的加权损失
            losses = criterion(logits, labels.to(device))
            # 取 batch 平均损失，用于反向传播
            loss = losses.mean()
            # 计算各个参数的梯度
            loss.backward()
            # 限制梯度范数以稳定 RNN 训练。
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 根据梯度，真正更新模型参数
            optimizer.step()
            total_train_loss += losses.detach().sum().item()

        # 4. 每轮结束后验证、记录并保留最佳参数
        probabilities, validation_loss = predict_model2(
            model, validation_loader, device, criterion
        )
        # 用阈值得到类别
        predictions = (probabilities >= 0.5).astype(int)
        # 计算 m-F1 等指标
        macro_f1 = f1_score(
            y_validation, predictions, labels=[0, 1],
            average="macro", zero_division=0,
        )
        train_loss = total_train_loss / len(train_dataset)
        # 将本轮的 training loss、validation loss、macro-F1 加入 history
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "validation_loss": validation_loss, "validation_macro_f1": macro_f1,
        })
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"Train loss: {train_loss:.4f} | Validation loss: {validation_loss:.4f} | "
            f"Validation macro-F1: {macro_f1:.4f}"
        )
        # 如果 macro-F1 超过此前最高分，保存当前参数的独立副本。
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            # clone 保存此时的独立副本，后续训练不会把它覆盖。
            best_state = {}
            for name, value in model.state_dict().items():
                best_state[name] = value.detach().cpu().clone()

    # 训练结束，恢复最佳模型并保存
    model.load_state_dict(best_state)
    probabilities, _ = predict_model2(model, validation_loader, device, criterion)
    print(f"Best epoch: {best_epoch}; validation macro-F1: {best_macro_f1:.4f}")
    # 将模型恢复成 validation macro-F1 最高那一轮的参数，再生成一次对应的 validation 概率。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 保存：.pt：最佳模型参数、词表和配置等
    #       .csv：每轮训练记录
    torch.save({
        "model_state_dict": best_state, "vocabulary": vocabulary,
        "config": config, "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "threshold": 0.5, "tokenization": "lower().split()",
        "class_weights": weights.tolist(),
    }, OUTPUT_DIR / "model2_bilstm.pt")
    history = pd.DataFrame(history)
    history.to_csv(OUTPUT_DIR / "model2_training_history.csv", index=False)
    # main() 可以拿最佳模型的预测，继续调用之前写好的整体评估、分语言评估和预测表整理函数。
    return model, vocabulary, probabilities, history


# 10. Main

def main():
    # ---------- Load shared data ----------
    train, validation = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y_train = train["answerable"].astype(int).to_numpy()
    y_validation = validation["answerable"].astype(int).to_numpy()
    comparison = []

    print("===== Data Statistics =====")
    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(validation)}")
    print(f"Languages: {LANGUAGES}")


    X_train, X_validation, feature_names, preprocessors = build_model1_features(train, validation)
    print("\n===== Model 1 Features =====")
    print(f"Training feature matrix: {X_train.shape}")
    print(f"Validation feature matrix: {X_validation.shape}")
    model1 = train_model1(X_train, y_train)
    probabilities1 = model1.predict_proba(X_validation)[:, 1]
    predictions1 = (probabilities1 >= 0.5).astype(int)
    evaluate_classifier(validation, predictions1, "Model 1: Logistic Regression")
    table1 = make_prediction_table(validation, predictions1, probabilities1)
    table1.to_csv(OUTPUT_DIR / "model1_predictions.csv", index=False, encoding="utf-8-sig")
    inspect_coefficients(model1, feature_names)
    comparison.append({
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_validation, predictions1),
        "macro_f1": f1_score(y_validation, predictions1, average="macro", zero_division=0),
    })

    model2, vocabulary, probabilities2, history = train_model2(
        train, validation, MODEL2_CONFIG
    )
    predictions2 = (probabilities2 >= 0.5).astype(int)
    # 直接复用你已有的整体/分语言 evaluation 和错误分析表。
    evaluate_classifier(validation, predictions2, "Model 2: BiLSTM")
    table2 = make_prediction_table(validation, predictions2, probabilities2)
    table2.to_csv(OUTPUT_DIR / "model2_predictions.csv", index=False, encoding="utf-8-sig")
    print("\n===== Model 2 Validation Predictions: First 10 =====")
    print(table2[[
        "example_id", "lang", "gold", "prediction", "p_answerable", "correct"
    ]].head(10).to_string(index=False))
    comparison.append({
        "model": "BiLSTM",
        "accuracy": accuracy_score(y_validation, predictions2),
        "macro_f1": f1_score(y_validation, predictions2, average="macro", zero_division=0),
    })

    comparison = pd.DataFrame(comparison)
    print("\n===== Overall Validation Comparison =====")
    print(comparison.to_string(index=False))
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()