import math
import pandas as pd
from collections import Counter
from datasets import load_from_disk

# 1. Configuration

DATA_PATH = r"D:\Python\nlp_data\data\tydi_xor_rc_ar_ko_te"

LANGUAGE = "ko"
MIN_COUNT = 2
K = 0.1

BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# 2. Load questions

def load_questions(data_path, language):
    """Load all questions in the selected language."""

    dataset = load_from_disk(data_path)

    train_questions = [
        row["question"]
        for row in dataset["train"]
        if row["lang"] == language
    ]

    validation_questions = [
        row["question"]
        for row in dataset["validation"]
        if row["lang"] == language
    ]

    return train_questions, validation_questions

# 3. Tokenisation

def tokenize_questions(questions):
    """
    question.split()
    Punctuation remains attached when there is no whitespace.
    """

    tokenized_questions = [
        question.split()
        for question in questions
    ]

    return tokenized_questions

# 4. Build vocabulary from TRAINING questions only

def build_vocab(train_tokens, min_count):
    # Count occurrences across all training questions. Keep tokens whose training frequency reaches min_count.

    token_counts = Counter(
        token # Yield each token to Counter.
        for question in train_tokens # Outer loop: iterate over questions.
        for token in question # Inner loop: iterate over the tokens in each question.
    )

    vocab = {
        token
        for token, count in token_counts.items()
        if count >= min_count
    }

    # Both are possible prediction targets, BOS is NOT, it is only used as history, never a prediction target.
    vocab.update({UNK, EOS})

    return vocab

# 5. Replace tokens outside the shared vocabulary

def replace_unknowns(tokenized_questions, vocab):
    """
    Apply the vocabulary to both splits.
    Training:
        Low-frequency tokens become UNK.
    Validation:
        Any token outside the training vocabulary becomes UNK.
        Validation counts never change the vocabulary.
    """
    result = []

    for question in tokenized_questions:
        new_question = []

        for token in question:
            if token in vocab:
                new_question.append(token)
            else:
                new_question.append(UNK)

        result.append(new_question)

    return result

# 6. Train the bigram model by counting

def train_bigram(train_tokens):

    # Count occurrences of each adjacent (previous, current) token pair.
    bigram_counts = Counter()
    # Count how often each token occurs as the previous token (the history).
    history_counts = Counter()

    for tokens in train_tokens:

        # Add boundaries for EACH question.
        sequence = [BOS] + tokens + [EOS]

        # Example:
        # [BOS, "what", "is", "AI", EOS]
        # Pairs:
        # (BOS, "what"), ("what", "is"), ("is", "AI"), ("AI", EOS)
        for i in range(len(sequence) - 1):
            previous = sequence[i]
            current = sequence[i + 1]
            bigram_counts[(previous, current)] += 1
            history_counts[previous] += 1

    return bigram_counts, history_counts

# 7. Add-k conditional probability

def bigram_probability(
    previous,
    current,
    bigram_counts,
    history_counts,
    vocab_size,
    k,
):
    """
    P(current | previous) = (C(previous, current) + k) / (C(previous) + k * vocab_size)
    Adding k to every candidate count gives unseen bigrams non-zero probability.
    """
    numerator = bigram_counts[(previous, current)] + k
    denominator = (history_counts[previous] + k * vocab_size)

    return numerator / denominator

# 8. Compute corpus-level per-token perplexity

def evaluate_perplexity(
    validation_tokens,
    bigram_counts,
    history_counts,
    vocab_size,
    k,
):
    """
    Score: Every question token, including UNK. One EOS per question.
    Do not score: BOS itself.

    Corpus PPL = exp(total negative log probability / total scored tokens)
    """

    total_nll = 0
    total_scored_tokens = 0

    for tokens in validation_tokens:
        sequence = [BOS] + tokens + [EOS]

        for i in range(len(sequence) - 1):
            previous = sequence[i]
            current = sequence[i + 1]
            probability = bigram_probability(
                previous=previous,
                current=current,
                bigram_counts=bigram_counts,
                history_counts=history_counts,
                vocab_size=vocab_size,
                k=k,
            )

            total_nll -= math.log(probability)
            total_scored_tokens += 1

    mean_nll = total_nll / total_scored_tokens
    perplexity = math.exp(mean_nll)

    return perplexity, mean_nll, total_scored_tokens

# 9. Compute question-level per-token perplexity

def score_questions(
    validation_questions,
    validation_tokens,
    bigram_counts,
    history_counts,
    vocab_size,
    k,
):

    results = []

    # Match each original question to its UNK-replaced tokens.
    for index in range(len(validation_questions)):
        question = validation_questions[index]
        tokens = validation_tokens[index]

        sequence = [BOS] + tokens + [EOS]

        # Reset the accumulated loss for each question.
        question_nll = 0

        for i in range(len(sequence) - 1):
            previous = sequence[i]
            current = sequence[i + 1]

            probability = bigram_probability(
                previous,
                current,
                bigram_counts,
                history_counts,
                vocab_size,
                k,
            )

            question_nll -= math.log(probability)

        # Score all text tokens plus one EOS, but not BOS.
        n_scored_tokens = len(tokens) + 1
        perplexity = math.exp(question_nll / n_scored_tokens)

        # UNK rate uses text tokens only, excluding boundaries.
        n_tokens = len(tokens)
        n_unk = tokens.count(UNK)
        unk_rate = n_unk / n_tokens

        results.append({
            "question_index": index,
            "question": question,
            "tokens_after_unk": tokens,
            "token_count": n_tokens,
            "unk_count": n_unk,
            "unk_rate": unk_rate,
            "perplexity": perplexity,
        })

    return pd.DataFrame(results)

# 10. Main workflow

def main():

    # ---------- Load questions ----------

    train_questions, validation_questions = load_questions(DATA_PATH, LANGUAGE)

    # ---------- Tokenise ----------

    train_tokens = tokenize_questions(train_questions)
    validation_tokens = tokenize_questions(validation_questions)

    # ---------- Build vocabulary list ----------

    vocab = build_vocab(train_tokens, MIN_COUNT)

    # ---------- Apply UNK mapping ----------

    train_tokens = replace_unknowns(train_tokens, vocab)
    validation_tokens = replace_unknowns(validation_tokens, vocab)

    # ---------- Train ----------

    bigram_counts, history_counts = train_bigram(train_tokens)

    # ---------- Overall Evaluate ----------

    perplexity, mean_nll, n_scored = evaluate_perplexity(
        validation_tokens=validation_tokens,
        bigram_counts=bigram_counts,
        history_counts=history_counts,
        vocab_size=len(vocab),
        k=K,
    )

    # Report UNK frequency to document how much text was mapped.
    n_validation_tokens = sum(len(tokens) for tokens in validation_tokens)
    n_validation_unk = sum(tokens.count(UNK) for tokens in validation_tokens)
    validation_unk_rate = n_validation_unk / n_validation_tokens

    # ---------- Score individual validation questions ----------

    question_scores = score_questions(
        validation_questions,
        validation_tokens,
        bigram_counts,
        history_counts,
        len(vocab),
        K,
    )

    # Index of the five questions with the highest perplexity and the five with the lowest perplexity.
    lowest = question_scores.sort_values(
        by=["perplexity", "question_index"],
        ascending=[True, True],
    ).head(5)

    # Highest perplexity first.
    highest = question_scores.sort_values(
        by=["perplexity", "question_index"],
        ascending=[False, True],
    ).head(5)

    formatters = {
        "unk_rate": "{:.2%}".format,
        "perplexity": "{:.4f}".format,
    }

    # ---------- Print results ----------

    print("===== Experiment setup =====")
    print(f"Language: {LANGUAGE}")
    print("Model: Bigram with add-k smoothing")
    print("Tokenisation: whitespace split")
    print(f"Minimum training token count: {MIN_COUNT}")
    print(f"Smoothing k: {K}")

    print("\n===== Data and vocabulary =====")
    print(f"Training questions: {len(train_questions)}")
    print(f"Validation questions: {len(validation_questions)}")
    print(f"Prediction vocabulary size: {len(vocab)}")
    print("Vocabulary includes UNK and EOS, but excludes BOS")
    print(f"Observed training bigrams: {len(bigram_counts)}")
    print(f"Validation text tokens: {n_validation_tokens}")
    print(f"Validation UNK tokens: {n_validation_unk}")
    print(f"Validation UNK rate: {validation_unk_rate:.2%}")

    print("\n===== Validation evaluation =====")
    print("Scoring: question tokens + EOS; BOS excluded")
    print(f"Total scored tokens: {n_scored}")
    print(f"Mean negative log probability: {mean_nll:.6f}")
    print(f"Per-token perplexity: {perplexity:.4f}")

    print("\n===== Lowest-perplexity questions =====")
    print(
        lowest.to_string(
            index=False,
            formatters=formatters,
        )
    )

    print("\n===== Highest-perplexity questions =====")
    print(
        highest.to_string(
            index=False,
            formatters=formatters,
        )
    )

    # ---------- Print the bigram training counts and predicted probabilities for
    # token pairs in the questions requiring further explanation and analysis. ----------

    for index in [24, 203, 19, 347, 73, 278, 163, 191]:
        print(f"\n===== Question {index} =====")
        print(validation_questions[index])

        tokens = validation_tokens[index]
        sequence = [BOS] + tokens + [EOS]

        rows = []

        for i in range(len(sequence) - 1):
            previous = sequence[i]
            current = sequence[i + 1]

            probability = bigram_probability(
                previous,
                current,
                bigram_counts,
                history_counts,
                len(vocab),
                K,
            )

            rows.append({
                "previous": previous,
                "current": current,
                "bigram_count": bigram_counts[(previous, current)],
                "history_count": history_counts[previous],
                "probability": probability,
            })

        print(
            pd.DataFrame(rows).to_string(
                index=False,
                formatters={"probability": "{:.6f}".format},
            )
        )

if __name__ == "__main__":
    main()

