
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "google-bert/bert-base-uncased"
DATASET_NAME = "amarfurt/art-nlp-hs26"
MAX_LENGTH = 128


class ARTDataset(Dataset):
    """Pre-tokenized, candidate-wise ART dataset."""

    def __init__(self, split, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        raw = load_dataset(DATASET_NAME, split=split)

        self.label_counts = dict(Counter(raw["label"]))

        if set(self.label_counts) != {1, 2}:
            raise ValueError(
                f"Unexpected labels in {split}: "
                f"{self.label_counts}"
            )

        self.samples = []

        for index, row in enumerate(raw):
            o1 = tokenizer.encode(
                row["observation_1"],
                add_special_tokens=False,
            )
            o2 = tokenizer.encode(
                row["observation_2"],
                add_special_tokens=False,
            )

            candidates = []

            for key in ("hypothesis_1", "hypothesis_2"):
                h = tokenizer.encode(
                    row[key],
                    add_special_tokens=False,
                )

                ids = (
                    [tokenizer.cls_token_id]
                    + o1
                    + [tokenizer.sep_token_id]
                    + h
                    + [tokenizer.sep_token_id]
                    + o2
                    + [tokenizer.sep_token_id]
                )

                if len(ids) > max_length:
                    raise ValueError(
                        f"{split} example {index}: "
                        f"{len(ids)} tokens > {max_length}"
                    )

                candidates.append(ids)

            self.samples.append({
                "input_ids": candidates,
                "label": int(row["label"]) - 1,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ARTCollator:
    """Dynamically pad both candidates in each batch."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_size = len(examples)

        max_len = max(
            len(ids)
            for example in examples
            for ids in example["input_ids"]
        )

        input_ids = torch.full(
            (batch_size, 2, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (batch_size, 2, max_len),
            dtype=torch.long,
        )

        labels = torch.empty(
            batch_size,
            dtype=torch.long,
        )

        for b, example in enumerate(examples):
            labels[b] = example["label"]

            for c, ids in enumerate(example["input_ids"]):
                length = len(ids)

                input_ids[b, c, :length] = torch.tensor(ids)
                attention_mask[b, c, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train = ARTDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    loader = DataLoader(
        train,
        batch_size=4,
        shuffle=False,
        collate_fn=ARTCollator(tokenizer.pad_token_id),
    )

    batch = next(iter(loader))

    print("Training examples:", len(train))
    print("Label distribution:", train.label_counts)

    print("Input shape:", batch["input_ids"].shape)
    print("Mask shape:", batch["attention_mask"].shape)
    print("Labels shape:", batch["labels"].shape)

    print("First labels:", batch["labels"].tolist())
    print(
        "First candidate lengths:",
        batch["attention_mask"][0].sum(dim=1).tolist(),
    )

    assert len(train) == 16801
    assert set(train.label_counts) == {1, 2}
    assert batch["input_ids"].shape[:2] == (4, 2)
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == (4,)
    assert batch["labels"][:3].tolist() == [1, 1, 0]
    assert batch["attention_mask"][0].sum(dim=1).tolist() == [32, 37]
    assert batch["input_ids"].shape[-1] <= MAX_LENGTH

    print("\nAll DataLoader checks passed!")
