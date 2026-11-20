
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
train = load_dataset("amarfurt/art-nlp-hs26", split="train")

lengths = []

for row in train:
    # No truncation: measure complete tokenized inputs
    o1 = tokenizer.encode(
        row["observation_1"], add_special_tokens=False
    )
    o2 = tokenizer.encode(
        row["observation_2"], add_special_tokens=False
    )

    for key in ("hypothesis_1", "hypothesis_2"):
        h = tokenizer.encode(
            row[key], add_special_tokens=False
        )

        # [CLS] O1 [SEP] H [SEP] O2 [SEP]
        length = len(o1) + len(h) + len(o2) + 4
        lengths.append(length)

lengths = np.asarray(lengths)

print("Dataset examples:", len(train))
print("Candidate sequences:", len(lengths))
print("Mean length:", round(lengths.mean(), 1))
print("Median:", int(np.median(lengths)))
print("95th percentile:", np.percentile(lengths, 95))
print("99th percentile:", np.percentile(lengths, 99))
print("Maximum:", lengths.max())

for limit in (128, 256, 384, 512):
    count = int((lengths > limit).sum())
    print(
        f"Longer than {limit}: {count} "
        f"({count / len(lengths) * 100:.2f}%)"
    )
