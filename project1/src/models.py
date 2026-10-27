import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        embedding_layer,
        pad_id,
        sep_id,
        hidden_dim=128,
        dropout=0.0
    ):
        super().__init__()

        # Pretrained frozen embeddings
        self.embedding = embedding_layer

        self.pad_id = pad_id
        self.sep_id = sep_id

        embedding_dim = embedding_layer.embedding_dim
        
        # Two-layer classifier with configurable dropout
        self.classifier = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ),
            nn.Linear(hidden_dim, 2)
        )


    def forward(self, input_ids, lengths=None):

        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedding(input_ids)

        # Identify the four segments using [SEP]
        sep_mask = input_ids == self.sep_id
        segment_ids = sep_mask.long().cumsum(dim=1)

        
        # Ignore padding and separators
        valid_mask = (
            (input_ids != self.pad_id) &
            (input_ids != self.sep_id)
        )

        # Also respect original sequence lengths
        if lengths is not None:
            positions = torch.arange(
                input_ids.size(1),
                device=input_ids.device
            ).unsqueeze(0)

            length_mask = (
                positions <
                lengths.to(input_ids.device).unsqueeze(1)
            )

            valid_mask = valid_mask & length_mask


        segment_vectors = []

        # Mean pooling for each segment
        for segment in range(4):

            mask = (
                (segment_ids == segment) & valid_mask
            ).unsqueeze(-1)

            summed = (embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)

            mean_vector = summed / counts
            segment_vectors.append(mean_vector)

        # Concatenate O1, O2, H1 and H2
        # [batch_size, 1200]
        x = torch.cat(segment_vectors, dim=1)

        # [batch_size, 2]
        logits = self.classifier(x)

        return logits


class EmbeddingGRU(nn.Module):
    """
    Two-layer GRU for ART classification.

    Uses frozen fastText embeddings and the
    final hidden state for classification.
    """

    def __init__(
        self,
        embedding_layer,
        pad_id,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        # Frozen pretrained embeddings
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad_(False)

        self.pad_id = pad_id

        embedding_dim = embedding_layer.embedding_dim

        # Two-layer GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Two-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input_ids, lengths=None):

        if lengths is None:
            lengths = (
                input_ids != self.pad_id
            ).sum(dim=1)

        # Embeddings: [batch, seq_len, 300]
        embeddings = self.embedding(input_ids)

        # pack_padded_sequence requires CPU lengths
        lengths_cpu = lengths.detach().to(
            device="cpu",
            dtype=torch.long
        )

        packed = pack_padded_sequence(
            embeddings,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False
        )

        # GRU final hidden states
        _, hidden = self.gru(packed)

        # Last layer hidden state: [batch, hidden_dim]
        final_hidden = hidden[-1]

        # Classification logits: [batch, 2]
        logits = self.classifier(final_hidden)

        return logits
