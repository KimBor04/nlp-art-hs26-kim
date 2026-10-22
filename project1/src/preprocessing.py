
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
    Tokenize ART text while preserving negation
    and representing possessive relationships.
    """
    text = text.lower()

    # Normalize apostrophe variants
    text = text.replace("’", "'").replace("´", "'")

    # Repair clearly corrupted apostrophes
    text = re.sub(r"(?<=\w)\uFFFD+(?=[st]\b)", "'", text)

    # Irregular negative contractions
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)

    # Regular negative contractions
    text = re.sub(r"\b([a-z]+)n't\b", r"\1 not", text)

    # Unambiguous positive contractions
    text = re.sub(r"\b([a-z]+)'m\b", r"\1 am", text)
    text = re.sub(r"\b([a-z]+)'re\b", r"\1 are", text)
    text = re.sub(r"\b([a-z]+)'ve\b", r"\1 have", text)
    text = re.sub(r"\b([a-z]+)'ll\b", r"\1 will", text)

    # Tokenize words and punctuation
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

    # Avoid interpreting ambiguous contractions as possessives
    ambiguous = {
        "it's", "he's", "she's", "that's",
        "there's", "here's", "what's",
        "who's", "where's", "how's",
        "let's"
    }

    result = []

    for token in tokens:
        if token.endswith("'s") and token not in ambiguous:
            result.extend([token[:-2], "possessive"])
        else:
            result.append(token)

    return result

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
