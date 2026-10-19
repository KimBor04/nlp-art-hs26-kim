
import numpy as np
import torch
import torch.nn as nn

from .preprocessing import PAD, UNK, SEP


def build_embedding_layer(vocab, fasttext, seed=42):
    """
    Create a frozen PyTorch embedding layer
    using pretrained fastText vectors.
    """

    embedding_dim = fasttext.vector_size

    # Initialize embedding matrix with zeros
    embedding_matrix = np.zeros(
        (len(vocab), embedding_dim),
        dtype=np.float32
    )

    # Copy pretrained word vectors
    for word, idx in vocab.items():
        if word in [PAD, UNK, SEP]:
            continue

        embedding_matrix[idx] = fasttext[word]

    # UNK: mean of known word vectors
    embedding_matrix[vocab[UNK]] = (
        embedding_matrix[3:].mean(axis=0)
    )

    # SEP: fixed random vector
    rng = np.random.default_rng(seed)
    embedding_matrix[vocab[SEP]] = rng.normal(
        0, 0.05, embedding_dim
    )

    # PAD remains zero

    # Create frozen embedding layer
    embedding_layer = nn.Embedding.from_pretrained(
        torch.from_numpy(embedding_matrix),
        freeze=True,
        padding_idx=vocab[PAD]
    )

    return embedding_layer
