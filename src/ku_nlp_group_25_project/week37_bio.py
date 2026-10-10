import re

from ku_nlp_group_25_project.utilities.load_dataset import load_dataset


def tokenize_with_offsets(text):
    pattern = r"\w+|[^\w\s]"
    matches = list(re.finditer(pattern, text))

    tokens = [match.group() for match in matches]
    offsets = [(match.start(), match.end()) for match in matches]

    return tokens, offsets

def span_to_bio(context, answerable, answer_start, answer):
    tokens, offsets = tokenize_with_offsets(context)

    # Start with every token outside the answer
    labels = ["O"] * len(tokens)

    # Unanswerable question -> everything stays O
    if not answerable:
        return tokens, offsets, labels

    answer_end = answer_start + len(answer)

    # Check that the dataset's answer really matches the context
    assert context[answer_start:answer_end] == answer

    # Find tokens that overlap with the answer span
    answer_token_indices = []

    for i, (token_start, token_end) in enumerate(offsets):
        if token_start < answer_end and token_end > answer_start:
            answer_token_indices.append(i)

    if not answer_token_indices:
        raise ValueError("No tokens found for answer span")

    # Check that the answer begins and ends exactly on token boundaries
    first_index = answer_token_indices[0]
    last_index = answer_token_indices[-1]

    if offsets[first_index][0] != answer_start:
        raise ValueError("Answer does not start on a token boundary")

    if offsets[last_index][1] != answer_end:
        raise ValueError("Answer does not end on a token boundary")

    # First answer token = B-ANS
    labels[first_index] = "B-ANS"

    # Remaining answer tokens = I-ANS
    for i in answer_token_indices[1:]:
        labels[i] = "I-ANS"

    return tokens, offsets, labels

def bio_to_span(context, offsets, labels):
    answer_indices = [
        i for i, label in enumerate(labels)
        if label in {"B-ANS", "I-ANS"}
    ]

    # No answer predicted
    if not answer_indices:
        return -1, ""

    # The first answer token must start with B-ANS
    first_index = answer_indices[0]

    if labels[first_index] != "B-ANS":
        raise ValueError("Answer span must start with B-ANS")

    # Check that the answer tokens are contiguous
    expected_indices = list(range(answer_indices[0], answer_indices[-1] + 1))

    if answer_indices != expected_indices:
        raise ValueError("Answer tokens must form one continuous span")

    # Everything after B-ANS must be I-ANS
    for i in answer_indices[1:]:
        if labels[i] != "I-ANS":
            raise ValueError("Invalid BIO sequence")

    answer_start = offsets[first_index][0]
    answer_end = offsets[answer_indices[-1]][1]

    answer = context[answer_start:answer_end]

    return answer_start, answer

training_df, validation_df = load_dataset()

# example = training_df[
#     (training_df["answerable"]) &
#     (training_df["answer"].str.contains(" ", na=False))
# ].iloc[0]

# example = training_df[
#     training_df["answerable"] == False
# ].iloc[0]
#
# tokens, offsets, labels = span_to_bio(
#     example["context"],
#     example["answerable"],
#     example["answer_start"],
#     example["answer"],
# )
#
# print("Question:")
# print(example["question"])
#
# print("\nAnswer:")
# print(example["answer"])
#
# print("\nAnswer start:")
# print(example["answer_start"])
#
# print("\nAnswer tokens:")
# for token, label in zip(tokens, labels):
#     if label != "O":
#         print(token, label)
#
# print("Question:")
# print(example["question"])
#
# print("\nAnswerable:")
# print(example["answerable"])
#
# print("\nUnique labels:")
# print(set(labels))

# example = training_df[
#     (training_df["answerable"]) &
#     (training_df["answer"].str.contains(" ", na=False))
# ].iloc[0]
#
# tokens, offsets, labels = span_to_bio(
#     example["context"],
#     example["answerable"],
#     example["answer_start"],
#     example["answer"],
# )
#
# recovered_start, recovered_answer = bio_to_span(
#     example["context"],
#     offsets,
#     labels,
# )
#
# print("Original answer:")
# print(example["answer"])
#
# print("\nRecovered answer:")
# print(recovered_answer)
#
# print("\nOriginal start:")
# print(example["answer_start"])
#
# print("\nRecovered start:")
# print(recovered_start)

# 5 answerable examples
answerable_examples = training_df[
    training_df["answerable"] == True
].sample(n=5, random_state=42)

# 5 unanswerable examples
unanswerable_examples = training_df[
    training_df["answerable"] == False
].sample(n=5, random_state=42)

print("\nANSWERABLE ROUND-TRIP TESTS")

for _, example in answerable_examples.iterrows():
    tokens, offsets, labels = span_to_bio(
        example["context"],
        example["answerable"],
        example["answer_start"],
        example["answer"],
    )

    recovered_start, recovered_answer = bio_to_span(
        example["context"],
        offsets,
        labels,
    )

    assert recovered_start == example["answer_start"]
    assert recovered_answer == example["answer"]

    print("PASS:", example["answer"])


print("\nUNANSWERABLE ROUND-TRIP TESTS")

for _, example in unanswerable_examples.iterrows():
    tokens, offsets, labels = span_to_bio(
        example["context"],
        example["answerable"],
        example["answer_start"],
        example["answer"],
    )

    recovered_start, recovered_answer = bio_to_span(
        example["context"],
        offsets,
        labels,
    )

    assert set(labels) == {"O"}
    assert recovered_start == -1
    assert recovered_answer == ""

    print("PASS: unanswerable")

print("\nBOUNDARY TEST: ANSWER STARTS AT 0")

context = "Paris is the capital of France."
answer = "Paris"
answer_start = 0

tokens, offsets, labels = span_to_bio(
    context,
    True,
    answer_start,
    answer,
)

recovered_start, recovered_answer = bio_to_span(
    context,
    offsets,
    labels,
)

assert recovered_start == 0
assert recovered_answer == "Paris"

print("PASS")

print("\nBOUNDARY TEST: PUNCTUATION NEXT TO ANSWER")

context = "The capital is Paris, France."
answer = "Paris"
answer_start = context.index(answer)

tokens, offsets, labels = span_to_bio(
    context,
    True,
    answer_start,
    answer,
)

recovered_start, recovered_answer = bio_to_span(
    context,
    offsets,
    labels,
)

assert recovered_start == answer_start
assert recovered_answer == "Paris"

print("PASS")

print("\nBOUNDARY TEST: MULTI-TOKEN ANSWER")

context = "The scientist was Wilhelm Röntgen."
answer = "Wilhelm Röntgen"
answer_start = context.index(answer)

tokens, offsets, labels = span_to_bio(
    context,
    True,
    answer_start,
    answer,
)

recovered_start, recovered_answer = bio_to_span(
    context,
    offsets,
    labels,
)

assert recovered_start == answer_start
assert recovered_answer == answer

answer_labels = [
    label
    for label in labels
    if label != "O"
]

assert answer_labels == ["B-ANS", "I-ANS"]

print("PASS")
