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

    # Assign sample IDs so that errors can be located later.
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
    # Build separate vocabularies and TF-IDF representations for questions and contexts.
    # min_df=2 retains tokens appearing in at least two training documents.

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

    # Fit both the vocabulary and IDF weights on training data only, then transform the texts.
    # Example: 6,335 questions and a hypothetical 1,000-token vocabulary produce a (6335, 1000) matrix.
    # Each row is a question; each column is a vocabulary token; each value is its TF-IDF weight.
    train_question_features = question_vectorizer.fit_transform(
        train["question"]
    )
    train_context_features = context_vectorizer.fit_transform(
        train["context"]
    )

    # Transform validation data using the fitted training vocabulary and IDF weights; do not refit.
    validation_question_features = question_vectorizer.transform(
        validation["question"]
    )
    validation_context_features = context_vectorizer.transform(
        validation["context"]
    )

    # Compute the numerical features.
    train_numeric = extract_numeric_features(train)
    validation_numeric = extract_numeric_features(validation)

    # Standardise numerical features: context lengths may be in the hundreds, while ratios range from 0 to 1.
    # For logistic regression, scaling helps optimisation converge and makes regularisation less sensitive to feature units.
    # As with TF-IDF, learn the means and standard deviations from training data only.
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train_numeric)
    validation_numeric = scaler.transform(validation_numeric)

    # Concatenate three feature blocks column-wise: [question TF-IDF | context TF-IDF | numerical features].
    # TF-IDF is sparse; convert the dense NumPy numerical block to sparse format before concatenation.
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

    # Keep feature names in the same order as the concatenated columns to interpret the coefficients correctly.
    feature_names = []

    for token in question_vectorizer.get_feature_names_out():
        feature_names.append(f"question::{token}")

    for token in context_vectorizer.get_feature_names_out():
        feature_names.append(f"context::{token}")

    feature_names.extend(NUMERIC_FEATURE_NAMES)

    # Keep the fitted preprocessors together so they can be reused for predictions on new data.
    preprocessors = {
        "question_vectorizer": question_vectorizer,
        "context_vectorizer": context_vectorizer,
        "scaler": scaler,
    }

    return X_train, X_validation, feature_names, preprocessors

# 3. Model 1: training

def train_model1(X_train, y_train):
    model = LogisticRegression(
    # C is the inverse regularisation strength; alternative values can be compared on validation data.
    # Balanced class weights give the minority unanswerable class greater weight; class_weight=None is a possible comparison.
    # The solver selects the optimisation algorithm used to estimate coefficients and the intercept.
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(X_train, y_train)
    return model

# 4. Shared evaluation for both models (！！！)
# Includes overall evaluation, evaluation by language, and prediction-table construction.

# Compute classification metrics from the model's predictions.
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
    # Report precision, recall, and F1 separately for both classes.
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["unanswerable", "answerable"],
        digits=4,
        zero_division=0,
    ))

# Compare predicted and gold labels on the full validation set, then separately for each language.
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

# Build a prediction table with labels and diagnostic fields for later error analysis.
def make_prediction_table(validation, predictions, probabilities):
    # probabilities contains the model's P(answerable).
    results = validation[
        ["example_id", "lang", "question", "context"]
    ].copy()

    results["gold"] = (
        validation["answerable"].astype(int).to_numpy()
    )
    results["prediction"] = predictions
    results["p_answerable"] = probabilities

    # Confidence is the probability assigned to the predicted class, not a separately calibrated correctness rate.
    # np.where(condition, true_value, false_value) selects P(answerable) for class 1, otherwise P(unanswerable).
    # Example: P(answerable)=0.90 gives predicted class 1 and confidence 0.90.
    #          P(answerable)=0.20 gives predicted class 0 and confidence 1-0.20=0.80.
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
        # model.coef_ has shape (1, n_features); [0] extracts the single row of learned weights.
        "coefficient": model.coef_[0],
    })

    # Inspect the largest and smallest coefficients: positive weights favour answerable, negative weights favour unanswerable.
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

    # The coefficient rankings include all three feature blocks.
    # Also print the four manual features (two lengths, shared-token count, and overlap ratio), even if absent from the top ten.
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
# IDs 0 and 1 are identifiers, not magnitudes or importance scores; ordinary token IDs start at 2.
PAD_ID = 0 # <PAD> has ID 0 and pads sequences of different lengths within a batch.
UNK_ID = 1 # <UNK> has ID 1 and represents tokens outside the vocabulary.
# Create the output directory.
# Save model1_predictions.csv, model2_predictions.csv, model2_training_history.csv, model2_bilstm.pt, and model_comparison.csv.
OUTPUT_DIR = Path(__file__).resolve().parent / "w39_outputs"


def set_seed(seed):
    # Three main sources of randomness in Model 2:
    # 1. Parameter initialisation: embedding, LSTM, and classifier parameters start from random values.
    # 2. Training order: shuffle=True shuffles examples for each epoch.
    # 3. Dropout: randomly sets some activations to zero during training.
    # Fix the seed to make these random processes repeatable as far as possible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Request deterministic cuDNN algorithms to improve reproducibility for the same inputs and settings.
    torch.backends.cudnn.deterministic = True
    # Disable runtime benchmarking so timing fluctuations do not change the selected algorithm between runs.
    torch.backends.cudnn.benchmark = False

# 7. Model 2: vocabulary, Dataset and batches
def build_vocabulary(train, min_freq):
    # Count tokens only in training questions and contexts.
    # Both text types share one vocabulary and reuse Model 1's lower().split() tokenisation.
    # min_freq refers to total token frequency, not document frequency.
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
    # Define a dataset class following the structure of ClassificationDatasetReader.
    # It stores processed examples and provides length and indexed access for the DataLoader.
    # Convert each pandas row into the format required for PyTorch training.
    # Each example becomes (question_token_ids, context_token_ids, answerable_label).
    def __init__(self, frame, vocabulary):
    # Process and store all examples.
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
    # Return the number of examples.
        return len(self.examples)

    def __getitem__(self, index):
    # Return the example at the requested index.
        return self.examples[index]


def collate_model2_batch(examples):
    # [Lab 4 adapted] Batch padding follows the role of collate_batch_transformer.
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
    # Build a recurrent classifier with a trainable embedding and two independent BiLSTM encoders.
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        # super() initialises nn.Module; assigning layers to self registers the embedding, encoders, and classifier.
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
        # Each BiLSTM produces 2 * hidden_dim features; concatenating question and context gives 4 * hidden_dim.
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, token_ids, lengths, lstm):
        # Encode a text sequence as one vector.
        # token_ids: [batch_size, sequence_length]
        # embedded:  [batch_size, sequence_length, embedding_dim]
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = lstm(packed)
        # For a single-layer BiLSTM, hidden[0] is the final forward state and hidden[1] the final backward state.
        return torch.cat([hidden[0], hidden[1]], dim=1)

    def forward(self, question_ids, context_ids, question_lengths, context_lengths):
        # Call encode twice to combine question and context encoding in the forward pass.
        question_vector = self.encode(
            question_ids, question_lengths, self.question_lstm
        )
        context_vector = self.encode(
            context_ids, context_lengths, self.context_lstm
        )
        combined = torch.cat([question_vector, context_vector], dim=1)
        # Output two unnormalised logits, ordered as [unanswerable, answerable].
        # CrossEntropyLoss applies log-softmax internally; pass logits without a separate softmax.
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
            # Column index 1 gives P(answerable), matching Model 1's probability convention.
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(batch_probabilities.cpu().tolist())
    return np.asarray(probabilities), total_loss / len(loader.dataset)


def train_model2(train, validation, config):
    # 1. Prepare the data and model.
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = build_vocabulary(train, config["min_freq"])
    train_dataset = AnswerabilityDataset(train, vocabulary)
    validation_dataset = AnswerabilityDataset(validation, vocabulary)

    # Shuffle training data only; keep validation order aligned with the original DataFrame for the prediction table.
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
    # 2. Set up the training objective and optimiser.
    # criterion measures the loss between predictions and gold labels.
    # optimizer updates model parameters using their gradients.
    # Initialise tracking for the best score, epoch, parameters, and per-epoch history.
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

    # 3. Train using nested epoch and batch loops.
    # The outer loop iterates over epochs; the inner loop processes every training batch once per epoch.
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        for questions, contexts, q_lengths, c_lengths, labels in train_loader:
            # Clear previous gradients to prevent unintended accumulation.
            optimizer.zero_grad()
            # Run the forward pass to obtain the two class logits.
            logits = model(
                questions.to(device), contexts.to(device), q_lengths, c_lengths
            )
            # Compute the class-weighted loss for each example.
            losses = criterion(logits, labels.to(device))
            # Average the batch losses for backpropagation.
            loss = losses.mean()
            # Backpropagate to compute parameter gradients.
            loss.backward()
            # Clip the gradient norm to stabilise recurrent training.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update model parameters using the gradients.
            optimizer.step()
            total_train_loss += losses.detach().sum().item()

        # 4. Validate after each epoch, record the results, and retain the best parameters.
        probabilities, validation_loss = predict_model2(
            model, validation_loader, device, criterion
        )
        # Apply the threshold to obtain predicted classes.
        predictions = (probabilities >= 0.5).astype(int)
        # Compute macro-F1 and the other metrics.
        macro_f1 = f1_score(
            y_validation, predictions, labels=[0, 1],
            average="macro", zero_division=0,
        )
        train_loss = total_train_loss / len(train_dataset)
        # Record training loss, validation loss, and macro-F1 in the history.
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "validation_loss": validation_loss, "validation_macro_f1": macro_f1,
        })
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"Train loss: {train_loss:.4f} | Validation loss: {validation_loss:.4f} | "
            f"Validation macro-F1: {macro_f1:.4f}"
        )
        # If macro-F1 improves, retain an independent copy of the current parameters.
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            # clone creates a separate copy that subsequent training cannot overwrite.
            best_state = {}
            for name, value in model.state_dict().items():
                best_state[name] = value.detach().cpu().clone()

    # After training, restore and save the best model.
    model.load_state_dict(best_state)
    probabilities, _ = predict_model2(model, validation_loader, device, criterion)
    print(f"Best epoch: {best_epoch}; validation macro-F1: {best_macro_f1:.4f}")
    # Reload the parameters with the highest validation macro-F1, then recompute their validation probabilities.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save the best parameters, vocabulary, and configuration in a .pt file.
    # Save the per-epoch training history as CSV.
    torch.save({
        "model_state_dict": best_state, "vocabulary": vocabulary,
        "config": config, "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "threshold": 0.5, "tokenization": "lower().split()",
        "class_weights": weights.tolist(),
    }, OUTPUT_DIR / "model2_bilstm.pt")
    history = pd.DataFrame(history)
    history.to_csv(OUTPUT_DIR / "model2_training_history.csv", index=False)
    # main() reuses the shared evaluation and prediction-table functions with the best model's predictions.
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
    # Reuse the existing overall/per-language evaluation and error-analysis table.
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