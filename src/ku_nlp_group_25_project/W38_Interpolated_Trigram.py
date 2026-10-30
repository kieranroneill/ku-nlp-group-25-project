from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

LANGUAGE = "ko"
MIN_COUNT = 2
K = 0.1

WEIGHTS = (0.15, 0.35, 0.50)

# Split each question into whitespace-separated tokens.
def tokenize_questions(questions):
    return [question.split() for question in questions]

# Build the vocabulary from training tokens that occur at least min_count times.
def build_vocab(train_tokens, min_count):
    counts = Counter(token for question in train_tokens for token in question)
    vocab = {token for token, count in counts.items() if count >= min_count}
    vocab.update({UNK, EOS})
    return vocab

# Replace tokens outside the training vocabulary with UNK.
def replace_unknowns(tokenised_questions, vocab):
    return [[token if token in vocab else UNK for token in question]
            for question in tokenised_questions]

# Count adjacent token pairs and their preceding-token frequencies.
def train_bigram(train_tokens):
    bigram_counts = Counter()
    history_counts = Counter()
    for tokens in train_tokens:
        sequence = [BOS] + tokens + [EOS]
        for i in range(len(sequence) - 1):
            previous, current = sequence[i], sequence[i + 1]
            bigram_counts[(previous, current)] += 1
            history_counts[previous] += 1
    return bigram_counts, history_counts

# Compute an add-k smoothed next-token probability for the bigram model.
def bigram_probability(previous, current, bigram_counts, history_counts,
                       vocab_size, k):
    return ((bigram_counts[(previous, current)] + k)
            / (history_counts[previous] + k * vocab_size))

# Load Korean training and validation questions from the shared dataset.
def prepare_questions():
    from ku_nlp_group_25_project.utilities.load_dataset import load_dataset
    train, validation = load_dataset()
    train_questions = train.loc[train["lang"] == LANGUAGE, "question"].tolist()
    validation_questions = validation.loc[
        validation["lang"] == LANGUAGE, "question"
    ].tolist()
    return train_questions, validation_questions

# Count training tokens and end-of-question markers.
def train_unigrams(train_tokens):
    counts = Counter()
    for tokens in train_tokens:
        counts.update(tokens)
        counts[EOS] += 1
    return counts

# Count next-token occurrences given the two preceding tokens.
def train_trigrams(train_tokens):
    trigram_counts = Counter()
    context_counts = Counter()
    for tokens in train_tokens:
        sequence = [BOS, BOS] + tokens + [EOS]
        for i in range(2, len(sequence)):
            context = (sequence[i - 2], sequence[i - 1])
            target = sequence[i]
            trigram_counts[(context[0], context[1], target)] += 1
            context_counts[context] += 1
    return trigram_counts, context_counts

# Compute an add-k smoothed unigram probability.
def unigram_probability(target, unigram_counts, vocab_size, k=K):
    total_targets = sum(unigram_counts.values())
    return (unigram_counts[target] + k) / (total_targets + k * vocab_size)

# Compute an add-k smoothed trigram probability.
def trigram_probability(history2, history1, target, trigram_counts,
                        context_counts, vocab_size, k=K):
    context = (history2, history1)
    return ((trigram_counts[(history2, history1, target)] + k)
            / (context_counts[context] + k * vocab_size))

# Combine unigram, bigram, and trigram probabilities using fixed weights.
def interpolated_probability(history2, history1, target, unigram_counts,
                             bigram_counts, bigram_history_counts,
                             trigram_counts, trigram_context_counts,
                             vocab_size, k=K, weights=WEIGHTS):
    unigram_weight, bigram_weight, trigram_weight = weights
    p1 = unigram_probability(target, unigram_counts, vocab_size, k)
    p2 = bigram_probability(history1, target, bigram_counts,
                            bigram_history_counts, vocab_size, k)
    p3 = trigram_probability(history2, history1, target, trigram_counts,
                             trigram_context_counts, vocab_size, k)
    return unigram_weight * p1 + bigram_weight * p2 + trigram_weight * p3

# Compute corpus perplexities and per-question evaluation results.
def evaluate_models(questions, encoded_questions, vocab,
                    unigram_counts, bigram_counts, bigram_history_counts,
                    trigram_counts, trigram_context_counts):
    total_nll = {name: 0.0 for name in ("unigram", "bigram", "trigram")}
    total_scored = 0
    rows = []
    vocab_size = len(vocab)

    for question_index, (question, tokens) in enumerate(zip(questions, encoded_questions)):
        sequence = [BOS, BOS] + tokens + [EOS]
        question_nll = {name: 0.0 for name in total_nll}
        for i in range(2, len(sequence)):
            h2, h1, target = sequence[i - 2], sequence[i - 1], sequence[i]
            p1 = unigram_probability(target, unigram_counts, vocab_size)
            p2 = bigram_probability(h1, target, bigram_counts,
                                    bigram_history_counts, vocab_size, K)
            p3 = interpolated_probability(
                h2, h1, target,
                unigram_counts, bigram_counts, bigram_history_counts,
                trigram_counts, trigram_context_counts, vocab_size,
            )
            for name, probability in (("unigram", p1), ("bigram", p2), ("trigram", p3)):
                assert 0 < probability <= 1.0, (name, probability)
                question_nll[name] -= math.log(probability)
            total_scored += 1

        n_scored_for_question = len(tokens) + 1
        for name in total_nll:
            total_nll[name] += question_nll[name]

        rows.append({
            "question_index": question_index,
            "question": question,
            "token_count": len(tokens),
            "unk_count": tokens.count(UNK),
            "unk_rate": tokens.count(UNK) / len(tokens) if tokens else 0.0,
            "unigram_ppl": math.exp(question_nll["unigram"] / n_scored_for_question),
            "bigram_ppl": math.exp(question_nll["bigram"] / n_scored_for_question),
            "trigram_ppl": math.exp(question_nll["trigram"] / n_scored_for_question),
        })

    if total_scored == 0:
        raise ValueError("No validation tokens were scored")
    perplexities = {
        name: math.exp(negative_log_likelihood / total_scored)
        for name, negative_log_likelihood in total_nll.items()
    }
    return perplexities, total_scored, rows

# Save question-level scores to a CSV file.
def save_csv(rows, output_file):
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} question scores to {path}")

# Print the questions with the lowest and highest perplexity.
def print_extremes(rows, key, count=5):
    for title, reverse in (("Lowest", False), ("Highest", True)):
        print(f"\n{title} {count} validation questions by {key}:")
        for row in sorted(rows, key=lambda r: (r[key] if not reverse else -r[key],
                                               r["question_index"]))[:count]:
            print(f"  #{row['question_index']:3d} PPL={row[key]:9.2f} "
                  f"UNK={row['unk_count']}/{row['token_count']} "
                  f"{row['question']}")

# Check probability normalisation and the scoring conventions.
def self_test():
    train_tokens = [["누가", "왔나요?"], ["누가", "갔나요?"]]
    vocab = set(["누가", "왔나요?", "갔나요?", UNK, EOS])
    unigram_counts = train_unigrams(train_tokens)
    bigram_counts, bigram_histories = train_bigram(train_tokens)
    trigram_counts, trigram_histories = train_trigrams(train_tokens)
    assert sum(unigram_counts.values()) == 6
    assert trigram_histories[(BOS, BOS)] == 2
    assert trigram_counts[(BOS, BOS, "누가")] == 2
    assert math.isclose(sum(WEIGHTS), 1.0)
    for h2, h1 in ((BOS, BOS), (BOS, "누가"), ("누가", "왔나요?"), (UNK, UNK)):
        summed = sum(interpolated_probability(
            h2, h1, target, unigram_counts, bigram_counts,
            bigram_histories, trigram_counts, trigram_histories, len(vocab),
        ) for target in vocab)
        assert math.isclose(summed, 1.0, abs_tol=1e-10), (h2, h1, summed)
    ppls, n_scored, rows = evaluate_models(
        ["누가 왔나요?"], [["누가", "왔나요?"]], vocab,
        unigram_counts, bigram_counts, bigram_histories,
        trigram_counts, trigram_histories,
    )
    assert n_scored == 3
    assert len(rows) == 1
    assert all(math.isfinite(value) and value > 0 for value in ppls.values())
    print("W38 self-tests passed (normalisation, BOS/EOS, token count, perplexity).")

# Run self-tests or train and evaluate the models on Korean questions.
def main():
    parser = argparse.ArgumentParser(description="Korean interpolated trigram language model")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", default="outputs/week_38/korean_lm_question_scores.csv")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    train_questions, validation_questions = prepare_questions()
    print(f"Korean questions: train={len(train_questions)}, val={len(validation_questions)}")
    train_tokenised = tokenize_questions(train_questions)
    validation_tokenised = tokenize_questions(validation_questions)
    vocab = build_vocab(train_tokenised, MIN_COUNT)
    train_tokens = replace_unknowns(train_tokenised, vocab)
    validation_tokens = replace_unknowns(validation_tokenised, vocab)

    unigram_counts = train_unigrams(train_tokens)
    bigram_counts, bigram_histories = train_bigram(train_tokens)
    trigram_counts, trigram_histories = train_trigrams(train_tokens)
    perplexities, n_scored, rows = evaluate_models(
        validation_questions, validation_tokens, vocab,
        unigram_counts, bigram_counts, bigram_histories,
        trigram_counts, trigram_histories,
    )
    n_val_tokens = sum(len(tokens) for tokens in validation_tokens)
    n_unks = sum(tokens.count(UNK) for tokens in validation_tokens)
    print(f"Vocabulary size (including UNK/EOS, excluding BOS): {len(vocab)}")
    print(f"Val UNK rate: {n_unks}/{n_val_tokens} = "
          f"{n_unks / n_val_tokens:.2%}" if n_val_tokens else "Val has no text tokens")
    print(f"Scored tokens (text + one EOS/question): {n_scored}")
    print(f"Add-k smoothing: k={K}")
    print(f"Interpolated trigram weights (uni, bi, tri): {WEIGHTS}")
    for model_name, ppl in perplexities.items():
        print(f"{model_name:8s} corpus validation PPL: {ppl:.4f}")
    print_extremes(rows, "trigram_ppl")
    save_csv(rows, args.output)

if __name__ == "__main__":
    main()
