
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from ku_nlp_group_25_project.utilities.load_dataset import load_dataset

MODEL_NAME = "FacebookAI/xlm-roberta-base"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using device:", device)

# Load the shared project dataset.
train, validation = load_dataset()

train = train[train["lang"] == "ko"].copy()
validation = validation[validation["lang"] == "ko"].copy()

print("Training examples:", len(train))
print("Validation examples:", len(validation))

# Load a pretrained multilingual tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

# Load XLM-RoBERTa with a question-answering prediction head.
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME,
)

model.to(device)

print("Tokenizer and model loaded successfully.")


# Convert a question-answer pair into model-ready inputs.
MAX_LENGTH = 384

def encode_qa_example(row):
    question = row["question"]
    context = row["context"]

    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_LENGTH,
        padding="max_length",
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]
    sequence_ids = encoded.sequence_ids()

    # For unanswerable questions, use the CLS token.
    cls_index = encoded["input_ids"].index(tokenizer.cls_token_id)

    start_position = cls_index
    end_position = cls_index

    if bool(row["answerable"]):
        answer_start = int(row["answer_start"])
        answer_end = answer_start + len(row["answer"])

        # Locate the context tokens containing the answer.
        context_positions = [
            i for i, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1 and offsets[i][1] > offsets[i][0]
        ]

        if context_positions:
            first = context_positions[0]
            last = context_positions[-1]

            # Assign an answer span only when it fits in the
            # tokenized context window.
            if (
                offsets[first][0] <= answer_start
                and offsets[last][1] >= answer_end
            ):
                for i in context_positions:
                    if offsets[i][0] <= answer_start < offsets[i][1]:
                        start_position = i
                        break

                for i in reversed(context_positions):
                    if offsets[i][0] < answer_end <= offsets[i][1]:
                        end_position = i
                        break

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "start_positions": start_position,
        "end_positions": end_position,
    }


print("Encoding training examples...")

train_features = [
    encode_qa_example(row)
    for _, row in train.iterrows()
]

validation_features = [
    encode_qa_example(row)
    for _, row in validation.iterrows()
]

print("Encoded training examples:", len(train_features))
print("Encoded validation examples:", len(validation_features))
print("First example start position:", train_features[0]["start_positions"])
print("First example end position:", train_features[0]["end_positions"])


from torch.utils.data import DataLoader

# Convert encoded examples into batches.
def collate_batch(batch):
    return {
        key: torch.tensor(
            [example[key] for example in batch],
            dtype=torch.long,
        )
        for key in (
            "input_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
        )
    }


train_loader = DataLoader(
    train_features,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_batch,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
)

EPOCHS = 2

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

# Save the trained model and tokenizer.
OUTPUT_DIR = "models/w37_xlmr_qa"

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
