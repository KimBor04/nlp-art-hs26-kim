
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "google-bert/bert-base-uncased"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL)
train = load_dataset("amarfurt/art-nlp-hs26", split="train")

required_fields = {
    "observation_1", "observation_2",
    "hypothesis_1", "hypothesis_2", "label"
}
assert required_fields.issubset(train.column_names)


def encode_candidate(row, hypothesis_key):
    o1 = tokenizer.encode(
        row["observation_1"], add_special_tokens=False
    )
    h = tokenizer.encode(
        row[hypothesis_key], add_special_tokens=False
    )
    o2 = tokenizer.encode(
        row["observation_2"], add_special_tokens=False
    )

    return (
        [tokenizer.cls_token_id]
        + o1
        + [tokenizer.sep_token_id]
        + h
        + [tokenizer.sep_token_id]
        + o2
        + [tokenizer.sep_token_id]
    )


print("Dataset fields:", train.column_names)
print("Label feature:", train.features["label"])

for i in range(3):
    row = train[i]

    ids1 = encode_candidate(row, "hypothesis_1")
    ids2 = encode_candidate(row, "hypothesis_2")

    assert row["label"] in (1, 2)
    assert len(ids1) <= MAX_LENGTH
    assert len(ids2) <= MAX_LENGTH

    target = row["label"] - 1

    print(f"\n=== Example {i} ===")
    print("Original label:", row["label"])
    print("PyTorch target:", target)
    print("Candidate lengths:", len(ids1), len(ids2))
    print("Candidate 1:", tokenizer.decode(ids1))
    print("Candidate 2:", tokenizer.decode(ids2))

print("\nInput formatting checks passed.")
