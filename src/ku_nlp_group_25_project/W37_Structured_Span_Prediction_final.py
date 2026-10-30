from datasets import load_from_disk
import numpy as np
import pandas as pd
import re

# load dataset
dataset = load_from_disk(
    r"D:\Python\nlp_data\data\tydi_xor_rc_ar_ko_te"
)

df_train = dataset["train"].to_pandas()
LANGUAGES = ["ar", "ko", "te"]
train = df_train[df_train["lang"].isin(LANGUAGES)]


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text):
    matches = list(TOKEN_PATTERN.finditer(text))
    tokens = [match.group(0) for match in matches]
    offsets = [(match.start(), match.end()) for match in matches]
    return tokens, offsets

# character span → BIO
def character_span_to_bio(context, answerable, answer_start=-1, answer_text=""):
    tokens, offsets = tokenize_with_offsets(context)
    labels = ["O"] * len(tokens)

# Adapt the Lab's no-answer check to use the dataset's answerable column.
    if not answerable:
        return tokens, offsets, labels

    answer_end = answer_start + len(answer_text)
    if context[answer_start:answer_end] != answer_text:
        raise ValueError("The supplied answer text does not match the character span")

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
    if not set(labels) <= {"O", "B-ANS", "I-ANS"}:
        raise ValueError("Only O, B-ANS and I-ANS labels are supported")

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


# Assert that the round trip recovers the original answer span.
# Sample five examples from each class.
answerable_samples = train[train["answerable"] == (True)].sample(n=5, random_state=42)
unanswerable_samples = train[train["answerable"] == (False)].sample(n=5, random_state=42)

samples = pd.concat([answerable_samples, unanswerable_samples])

for index, row in samples.iterrows():
    context = row["context"]
    answerable = row["answerable"]
    answer_start = row["answer_start"]
    answer_text = row["answer"]

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

    table = pd.DataFrame({"token": tokens, "offset": offsets, "label": labels,})
    print(table.to_string(index=False))

# Print the check result.
print("\nAll 10 round-trip checks passed.")



