from ku_nlp_group_25_project.utilities.load_dataset import load_dataset

training_df, validation_df = load_dataset()

print("Training examples:", len(training_df))
print("Validation examples:", len(validation_df))
print(training_df.head())

print("\nTraining class counts:")
print(training_df["answerable"].value_counts())

print("\nValidation class counts:")
print(validation_df["answerable"].value_counts())

print("\nTraining class percentages:")
print(training_df["answerable"].value_counts(normalize=True) * 100)

print("\nValidation class percentages:")
print(validation_df["answerable"].value_counts(normalize=True) * 100)

for split_name, df in [("Train", training_df), ("Validation", validation_df)]:
    print(f"\n{split_name} per-language counts:")

    for lang in ["ar", "ko", "te"]:
        lang_df = df[df["lang"] == lang]

        total = len(lang_df)
        answerable = lang_df["answerable"].sum()
        unanswerable = total - answerable
        unanswerable_pct = (unanswerable / total) * 100

        print(
            lang,
            "Total:", total,
            "Answerable:", answerable,
            "Unanswerable:", unanswerable,
            "Unanswerable %:", round(unanswerable_pct, 2)
        )
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# Find the majority class using TRAINING data only
majority_class = training_df["answerable"].mode()[0]

# Predict that class for every validation example
y_true = validation_df["answerable"]
y_pred = [majority_class] * len(validation_df)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=[True, False],
    zero_division=0,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

print("\nMajority-class baseline")
print("Majority class:", majority_class)
print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))

print("\nAnswerable")
print("Precision:", round(precision[0], 4))
print("Recall:", round(recall[0], 4))
print("F1:", round(f1[0], 4))

print("\nUnanswerable")
print("Precision:", round(precision[1], 4))
print("Recall:", round(recall[1], 4))
print("F1:", round(f1[1], 4))

for lang in ["ar", "ko", "te"]:
    lang_df = validation_df[validation_df["lang"] == lang]

    y_true_lang = lang_df["answerable"]
    y_pred_lang = [majority_class] * len(lang_df)

    accuracy_lang = accuracy_score(y_true_lang, y_pred_lang)
    macro_f1_lang = f1_score(
        y_true_lang,
        y_pred_lang,
        average="macro"
    )

    precision_lang, recall_lang, f1_lang, _ = precision_recall_fscore_support(
        y_true_lang,
        y_pred_lang,
        labels=[True, False],
        zero_division=0,
    )

    print(f"\nLanguage: {lang}")
    print("Accuracy:", round(accuracy_lang, 4))
    print("Macro-F1:", round(macro_f1_lang, 4))

    print("Answerable F1:", round(f1_lang[0], 4))
    print("Unanswerable F1:", round(f1_lang[1], 4))
