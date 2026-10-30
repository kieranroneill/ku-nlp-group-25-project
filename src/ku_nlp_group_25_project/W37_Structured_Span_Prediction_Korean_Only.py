import pandas as pd
import re
from ku_nlp_group_25_project.utilities.load_dataset import load_dataset

# Load the group's shared dataset
train, validation = load_dataset()

# Select Korean examples for our Week 37 work
train = train[train["lang"] == "ko"].copy()
validation = validation[validation["lang"] == "ko"].copy()
print(train[["lang", "question", "answer"]].head(5).to_string(index=False))
# ===========================================================================================
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Build a token vocabulary using only the training examples.
def build_vocabulary(examples):
    vocabulary = {"<PAD>": 0, "<UNK>": 1}

    for example in examples:
        for token in example["tokens"]:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary

LABEL_TO_ID = {"O": 0, "B-ANS": 1, "I-ANS": 2}

# Convert the tokens and BIO labels into numerical IDs
def encode_example(example, vocabulary):
    token_ids = [
        vocabulary.get(token, vocabulary["<UNK>"])
        for token in example["tokens"]
    ]

    label_ids = [
        LABEL_TO_ID[label]
        for label in example["labels"]
    ]

    return {
        **example,
        "question_ids": encode_question(example["question"], vocabulary),
        "token_ids": token_ids,
        "label_ids": label_ids,
    }
# Tokenize the question and convert its tokens into numerical IDs.
def encode_question(question, vocabulary):
    tokens, _ = tokenize_with_offsets(question)

    return [
        vocabulary.get(token, vocabulary["<UNK>"])
        for token in tokens
    ]

def tokenize_with_offsets(text):
    matches = list(TOKEN_PATTERN.finditer(text))
    tokens = [match.group(0) for match in matches]
    offsets = [(match.start(), match.end()) for match in matches]
    return tokens, offsets

#Split tokens at answer boundaries while preserving character offsets.
def align_tokens_to_answer(context, offsets, answer_start, answer_end):
    aligned_offsets = []

    for start, end in offsets:
        boundaries = [start, end]

        if start < answer_start < end:
            boundaries.append(answer_start)

        if start < answer_end < end:
            boundaries.append(answer_end)

        boundaries.sort()

        for left, right in zip(boundaries, boundaries[1:]):
            aligned_offsets.append((left, right))

    tokens = [context[start:end] for start, end in aligned_offsets]

    return tokens, aligned_offsets

# character span → BIO
def character_span_to_bio(context, answerable, answer_start=-1, answer_text=""):
    tokens, offsets = tokenize_with_offsets(context)
    labels = ["O"] * len(tokens)

# Lab中数据的情况和project里不一样，不能沿用判定方法，要改成用answerable列判断。
    if not answerable:
        return tokens, offsets, labels

    answer_end = answer_start + len(answer_text)
    if context[answer_start:answer_end] != answer_text:
        raise ValueError("The supplied answer text does not match the character span")

    tokens, offsets = align_tokens_to_answer(
        context, offsets, answer_start, answer_end
    )
    labels = ["O"] * len(tokens)

    covered = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < answer_end and end > answer_start
    ]
    if not covered:
        raise ValueError("The answer does not overlap any token")
    if offsets[covered[0]][0] != answer_start or offsets[covered[-1]][1] != answer_end:
        raise ValueError("The answer span does not align with token boundaries")

    labels[covered[0]] = "B-ANS"
    for index in covered[1:]:
        labels[index] = "I-ANS"
    return tokens, offsets, labels

# BIO → character span
def bio_to_character_span(context, offsets, labels):
    if len(offsets) != len(labels):
        raise ValueError("Offsets and labels must have the same length")
    if not validate_bio_sequence(labels):
        raise ValueError("Invalid BIO sequence: expected at most one contiguous answer span")

    answer_indices = [
        index for index, label in enumerate(labels) if label != "O"
    ]
    if not answer_indices:
        return -1, ""

    start_index = answer_indices[0]
    expected_indices = list(range(start_index, start_index + len(answer_indices)))
    expected_labels = ["B-ANS"] + ["I-ANS"] * (len(answer_indices) - 1)
    if answer_indices != expected_indices:
        raise ValueError("The labels contain multiple or non-contiguous answer spans")
    if [labels[index] for index in answer_indices] != expected_labels:
        raise ValueError("The answer span must begin with B-ANS and continue with I-ANS")

    start = offsets[answer_indices[0]][0]
    end = offsets[answer_indices[-1]][1]
    return start, context[start:end]

# Validate BIO labels for a single answer span.
def validate_bio_sequence(labels):
    allowed = {"O", "B-ANS", "I-ANS"}

    if any(label not in allowed for label in labels):
        return False

    inside_answer = False
    completed_answer = False

    for label in labels:
        if label == "B-ANS":
            if inside_answer or completed_answer:
                return False
            inside_answer = True

        elif label == "I-ANS":
            if not inside_answer:
                return False

        else:
            if inside_answer:
                completed_answer = True
                inside_answer = False

    return True
def test_bio_validator():

    # Valid: one answer span or no answer
    assert validate_bio_sequence(["O", "B-ANS", "I-ANS", "O"])
    assert validate_bio_sequence(["B-ANS", "O", "O"])
    assert validate_bio_sequence(["O", "O", "O"])

    # Invalid: missing beginning, interrupted span, or multiple spans
    assert not validate_bio_sequence(["O", "I-ANS", "O"])
    assert not validate_bio_sequence(["B-ANS", "O", "I-ANS"])
    assert not validate_bio_sequence(["B-ANS", "O", "B-ANS"])
    assert not validate_bio_sequence(["B-ANS", "INVALID"])

    print("BIO validator tests passed.")

test_bio_validator()

def test_span_boundaries():
    cases = [
        # Answer begins at character position zero
        ("New York City is busy.", True, "New York", 0),

        # Answer next to punctuation
        ("She lives in (Ulm), Germany.", True, "Ulm", None),

        # Multi-token answer
        ("The ship crossed the North Atlantic Ocean.", True,
         "North Atlantic Ocean", None),

        # Unanswerable example
        ("There is no answer here.", False, "", -1),
    ]

    for context, answerable, answer, start in cases:
        if answerable and start is None:
            start = context.index(answer)

        tokens, offsets, labels = character_span_to_bio(
            context, answerable, start, answer
        )

        recovered_start, recovered_answer = bio_to_character_span(
            context, offsets, labels
        )

        assert recovered_start == start
        assert recovered_answer == answer

        if answerable:
            assert labels.count("B-ANS") == 1
        else:
            assert all(label == "O" for label in labels)

    print("Boundary tests passed!")


test_span_boundaries()

# Prepare one QA example for the structured span prediction pipeline.
def prepare_example(row):
    context = row["context"]
    tokens, offsets, labels = character_span_to_bio(
        context,
        row["answerable"],
        row["answer_start"],
        row["answer"],
    )

    return {
        "question": row["question"],
        "context": context,
        "tokens": tokens,
        "offsets": offsets,
        "labels": labels,
        "answerable": row["answerable"],
    }

# Prepare all Korean examples for the span prediction pipeline.
train_examples = [prepare_example(row) for _, row in train.iterrows()]
validation_examples = [prepare_example(row) for _, row in validation.iterrows()]

print(f"Prepared {len(train_examples)} training examples.")
print(f"Prepared {len(validation_examples)} validation examples.")

vocabulary = build_vocabulary(train_examples)
print(f"Vocabulary size: {len(vocabulary)}")

encoded_train = [
    encode_example(example, vocabulary)
    for example in train_examples
]

encoded_validation = [
    encode_example(example, vocabulary)
    for example in validation_examples
]

print(f"Encoded {len(encoded_train)} training examples.")
print(f"Encoded {len(encoded_validation)} validation examples.")

# 检查assert 完全恢复原 answer
# Sample five examples from each class.
# answerable_samples = train[train["answerable"] == (True)].sample(n=5, random_state=42)
# unanswerable_samples = train[train["answerable"] == (False)].sample(n=5, random_state=42)

answerable_samples = train[train["answerable"] == True]
unanswerable_samples = train[train["answerable"] == False]

# samples = pd.concat([answerable_samples, unanswerable_samples])
samples = validation.copy()

alignment_fixes = 0
for index, row in samples.iterrows():
    context = row["context"]
    answerable = row["answerable"]
    answer_start = row["answer_start"]
    answer_text = row["answer"]

    if answerable:
        answer_end = answer_start + len(answer_text)
        _, original_offsets = tokenize_with_offsets(context)

        boundaries = {position for start, end in original_offsets for position in (start, end)}

        if answer_start not in boundaries or answer_end not in boundaries:
            alignment_fixes += 1

    print(f"Testing index {index}: answer={answer_text!r}, start={answer_start}")
    print("Before answer:", repr(context[max(0, answer_start - 15):answer_start]))
    print("After answer:", repr(context[answer_start + len(answer_text):answer_start + len(answer_text) + 15]))

    tokens, offsets, labels = character_span_to_bio(
        context, answerable, answer_start, answer_text
    )

    round_trip_start, round_trip_text = bio_to_character_span(
        context, offsets, labels
    )
    if answerable:
        assert round_trip_start == answer_start
        assert round_trip_text == answer_text
    else:
        assert all(label == "O" for label in labels)
        assert round_trip_start == -1
        assert round_trip_text == ""


    # Print every examples.
    print(
        f"\n Index: {index} "
        f"Answerable: {answerable} Check passed "
    )

    # table = pd.DataFrame({"token": tokens, "offset": offsets, "label": labels,})
    # print(table.to_string(index=False))

# Print the check result.
# print("\nAll 10 round-trip checks passed.")
print(f"\nAll {len(samples)} round-trip checks passed.")
# ===========================================================================================
print(f"Examples requiring token alignment: {alignment_fixes}")
print(f"Examples already aligned: {len(samples[samples['answerable'] == True]) - alignment_fixes}")


