
import re
from collections import Counter

# Dataset fields
FIELDS = [
    "observation_1",
    "observation_2",
    "hypothesis_1",
    "hypothesis_2"
]

# Special tokens
PAD = "[PAD]"
UNK = "[UNK]"
SEP = "[SEP]"


def tokenize(text):
    """
    Tokenize text while preserving important negations.
    """
    text = text.lower().replace("’", "'")

    # Irregular negative contractions
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)

    # Regular negative contractions
    text = re.sub(r"\b([a-z]+)n't\b", r"\1 not", text)

    # Tokenize words and punctuation
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def prepare_tokens(example):
    """
    Combine the four ART texts using [SEP] tokens.
    """
    tokens = []

    for i, field in enumerate(FIELDS):
        if i > 0:
            tokens.append(SEP)

        tokens.extend(tokenize(example[field]))

    return tokens

def build_vocab(train_dataset, fasttext):
    """
    Build vocabulary from training data only.
    Include only words covered by pretrained fastText.
    """
    word_counts = Counter()

    for example in train_dataset:
        for field in FIELDS:
            word_counts.update(tokenize(example[field]))

    # Special tokens
    vocab = {
        PAD: 0,
        UNK: 1,
        SEP: 2
    }

    # Add words with pretrained embeddings
    for word, count in word_counts.most_common():
        if word in fasttext.key_to_index:
            vocab[word] = len(vocab)

    return vocab


def encode_example(example, vocab):
    """
    Convert an ART example into token IDs and a label.
    """
    tokens = prepare_tokens(example)

    input_ids = [
        vocab.get(token, vocab[UNK])
        for token in tokens
    ]

    # Labels: 1 -> 0, 2 -> 1
    label = example["label"] - 1

    return {
        "input_ids": input_ids,
        "label": label
    }
