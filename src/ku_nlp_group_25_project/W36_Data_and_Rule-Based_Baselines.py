from datasets import load_from_disk
import numpy as np
import pandas as pd

# load dataset
dataset = load_from_disk(
    r"D:\Python\nlp_data\data\tydi_xor_rc_ar_ko_te"
)

# Print the sample counts and column labels for each split.
print(dataset)

# item 1 ==============================================================================================================
# Data preparation: Group by splits and languages.
df_train = dataset["train"].to_pandas()
df_val = dataset["validation"].to_pandas()

LANGUAGES = ["ar", "ko", "te"]

frames = {
    "train": df_train[df_train["lang"].isin(LANGUAGES)].copy(),
    "validation": df_val[df_val["lang"].isin(LANGUAGES)].copy(),
}


def token_count(text):
    """Whitespace tokenisation；Missing text is retained as missing, not as 0."""
    if pd.isna(text):
        return float("nan")
    return len(text.split())


def length_statistics(lengths, prefix):
    q1 = lengths.quantile(0.25)
    median = lengths.median()
    q3 = lengths.quantile(0.75)

    return {
        f"{prefix}_q1": q1,
        f"{prefix}_median": median,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": q3 - q1,
    }


summary_rows = []
missing_rows = []

for split, df in frames.items():
    for lang, group in df.groupby("lang", sort=True):
        n = len(group)

        # 确保标签有效，避免缺失标签被当成 unanswerable。
        valid_labels = group["answerable"].isin([True, False])
        if not valid_labels.all():
            raise ValueError(
                f"{split}/{lang}: missing or invalid values in 'answerable'"
            )

        n_answerable = int((group["answerable"] == (True)).sum())
        n_unanswerable = int((group["answerable"] == (False)).sum())

        # 为每条文本计算 token 数。
        question_lengths = group["question"].map(token_count)
        context_lengths = group["context"].map(token_count)

        # 在每组内检查完全相同的 question-context pairs。
        # 缺少 question/context 的行单独在缺失表报告，不将它们当作有效 pair 参与重复检查。
        pairs = group.dropna(subset=["question", "context"])

        duplicate_extra = int(
            pairs.duplicated(
                subset=["question", "context"],
                keep="first",
            ).sum()
        )

        duplicate_all = int(
            pairs.duplicated(
                subset=["question", "context"],
                keep=False,
            ).sum()
        )

        row = {
            "split": split,
            "lang": lang,
            "n_examples": n,
            "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable,
            "answerable_pct": 100 * n_answerable / n,
            "unanswerable_pct": 100 * n_unanswerable / n,
            "duplicate_extra_rows": duplicate_extra,
            "duplicate_all_rows": duplicate_all,
        }

        row.update(length_statistics(question_lengths, "question"))
        row.update(length_statistics(context_lengths, "context"))
        summary_rows.append(row)

        # 检查全部原始字段；缺失值和空白字符串分别统计。
        for column in group.columns:
            values = group[column]

            def is_blank_string(value):
                if isinstance(value, str):
                    return value.strip() == ""
                return False
            blank_count = values.map(is_blank_string).sum()

            missing_rows.append({
                "split": split,
                "lang": lang,
                "column": column,
                "missing_count": int(values.isna().sum()),
                "blank_string_count": int(blank_count),
            })


summary = pd.DataFrame(summary_rows)
missing = pd.DataFrame(missing_rows)

# 每个 split 的总体数量：将三种语言的计数相加。
count_columns = [
    "n_examples",
    "n_answerable",
    "n_unanswerable",
]

overall = summary.groupby("split")[count_columns].sum()

# 两个 split 合计，仅用于总体描述。
overall.loc["all"] = overall.sum()

# 用合计数量重新计算比例。
overall["answerable_pct"] = (
    100 * overall["n_answerable"] / overall["n_examples"]
)
overall["unanswerable_pct"] = (
    100 * overall["n_unanswerable"] / overall["n_examples"]
)

print("\n===== Statistics by Split and Language =====")
print(summary.round(2).to_string(index=False))

print("\n===== Overall Counts and Class Proportions =====")
print(overall.round(2).to_string())

print("\n===== Missing Values and Blank Strings by Column =====")
print(missing.to_string(index=False))

# item 2 ==============================================================================================================
top_token_rows = []

train = df_train[df_train["lang"].isin(LANGUAGES)].copy()

for lang, group in train.groupby("lang", sort=True):

    # 1. Collect all tokens from this language's training questions.
    all_tokens = []

    for question in group["question"].dropna():
        all_tokens.extend(question.split())

    # 2. Find unique tokens.
    unique_tokens = pd.Series(all_tokens, dtype="string").unique()

    # 3. Count occurrences of each token.
    token_counts = {}

    for token in unique_tokens:
        token_counts[token] = all_tokens.count(token)

    # 4. Sort by count in descending order and select the top five.
    top_five = sorted(
        token_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    # 5. Collect results.
    for rank, (token, count) in enumerate(top_five, start=1):
        top_token_rows.append({
            "lang": lang,
            "rank": rank,
            "token": token,
            "count": count,
        })

top_tokens = pd.DataFrame(top_token_rows)

print("\n===== Top 5 Training Question Tokens by Language =====")
print(top_tokens.to_string(index=False))

# item 3 ==============================================================================================================
results = []

for split, df in frames.items():

    # Select answerable examples in the required languages.
    selected = df[df["answerable"] == (True)]

    n_failed = 0

    for _, row in selected.iterrows():
        start = row["answer_start"]
        answer = row["answer"]
        context = row["context"]

        if context[start:start + len(answer)] != answer:
            n_failed += 1

    results.append({
        "split": split,
        "n_checked": len(selected),
        "n_failed": n_failed,
    })

summary = pd.DataFrame(results)

print("\n===== Answer Span Verification =====")
print(summary.to_string(index=False))

print(f"\nTotal checked: {summary['n_checked'].sum()}")
print(f"Total failed: {summary['n_failed'].sum()}")


# Baseline 1 ==========================================================================================================
train = df_train[df_train["lang"].isin(LANGUAGES)]
validation = df_val[df_val["lang"].isin(LANGUAGES)].copy()

# 1. Determine the majority class from training data only.
n_answerable = (train["answerable"] == (True)).sum()
n_unanswerable = (train["answerable"] == (False)).sum()

majority_class = bool(n_answerable >= n_unanswerable)

# 2. Predict the same class for every validation example.
validation["prediction"] = majority_class

print(f"Majority class: {majority_class}")

# 3. Calculate precision, recall and F1 for a given class.
def class_metrics(y_true, y_pred, label):
    tp = ((y_true == label) & (y_pred == label)).sum()
    fp = ((y_true != label) & (y_pred == label)).sum()
    fn = ((y_true == label) & (y_pred != label)).sum()

    # Use 0 when a denominator is zero.
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


# 4. Evaluate overall and separately for each language.
evaluation_groups = [("overall", validation)]
evaluation_groups += list(validation.groupby("lang", sort=True))

results = []

for name, group in evaluation_groups:
    y_true = group["answerable"]
    y_pred = group["prediction"]

    a_precision, a_recall, a_f1 = class_metrics(
        y_true, y_pred, label=True
    )

    u_precision, u_recall, u_f1 = class_metrics(
        y_true, y_pred, label=False
    )

    results.append({
        "group": name,
        "n_examples": len(group),
        "macro_f1": (a_f1 + u_f1) / 2,
        "answerable_precision": a_precision,
        "answerable_recall": a_recall,
        "answerable_f1": a_f1,
        "unanswerable_precision": u_precision,
        "unanswerable_recall": u_recall,
        "unanswerable_f1": u_f1,
    })

evaluation = pd.DataFrame(results)

print("\n===== Majority-Class Baseline: Validation Evaluation =====")
print(evaluation.round(4).to_string(index=False))







