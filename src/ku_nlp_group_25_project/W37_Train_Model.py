
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from ku_nlp_group_25_project.utilities.load_dataset import load_dataset

# Configuration
MODEL_NAME = "FacebookAI/xlm-roberta-base"
OUTPUT_DIR = "models/w37_xlmr_bio"
MAX_LENGTH = 384
BATCH_SIZE = 4
EPOCHS = 2

LABEL_TO_ID = {"O": 0, "B-ANS": 1, "I-ANS": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

# Load the same Korean dataset
train, validation = load_dataset()
train = train[train["lang"] == "ko"].copy()
validation = validation[validation["lang"] == "ko"].copy()

print("Training examples:", len(train))
print("Validation examples:", len(validation))

# Load multilingual tokenizer and BIO classification model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
).to(device)

print("BIO model loaded successfully.")


# Convert question and context into token-level BIO labels
def encode_bio_example(row):
    question = row["question"]
    context = row["context"]

    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]
    sequence_ids = encoded.sequence_ids()

    # Ignore question, special tokens, and padding in the loss
    labels = [
        LABEL_TO_ID["O"] if sid == 1 and end > start else -100
        for sid, (start, end) in zip(sequence_ids, offsets)
    ]

    if bool(row["answerable"]):
        answer_start = int(row["answer_start"])
        answer_text = row["answer"]
        answer_end = answer_start + len(answer_text)

        if context[answer_start:answer_end] != answer_text:
            raise ValueError("Answer annotation does not match context")

        context_indices = [
            i for i, sid in enumerate(sequence_ids)
            if sid == 1 and offsets[i][1] > offsets[i][0]
        ]

        # Skip answerable examples where the answer is truncated.
        if not context_indices:
            return None

        first_offset = offsets[context_indices[0]][0]
        last_offset = offsets[context_indices[-1]][1]

        if answer_start < first_offset or answer_end > last_offset:
            return None

        # Find context tokens that overlap the gold answer
        answer_indices = [
            i for i in context_indices
            if offsets[i][0] < answer_end
            and offsets[i][1] > answer_start
        ]

        if not answer_indices:
            return None

        labels[answer_indices[0]] = LABEL_TO_ID["B-ANS"]

        for i in answer_indices[1:]:
            labels[i] = LABEL_TO_ID["I-ANS"]

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


# Prepare training examples
print("Encoding BIO training examples...")

train_features = [
    encode_bio_example(row)
    for _, row in train.iterrows()
]

train_features = [
    example for example in train_features
    if example is not None
]

print("Usable training examples:", len(train_features))
print(
    "Skipped truncated-answer examples:",
    len(train) - len(train_features),
)


# Batch the examples
def collate_batch(batch):
    return {
        key: torch.tensor(
            [example[key] for example in batch],
            dtype=torch.long,
        )
        for key in ("input_ids", "attention_mask", "labels")
    }


train_loader = DataLoader(
    train_features,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
)

# Train the BIO model
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_number, batch in enumerate(train_loader, start=1):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_number % 50 == 0:
            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_number}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}",
                flush=True,
            )

    print(
        f"Epoch {epoch + 1} finished. "
        f"Average loss: {total_loss / len(train_loader):.4f}",
        flush=True,
    )

# Save the BIO model separately from the previous QA model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"BIO model saved to {OUTPUT_DIR}")
