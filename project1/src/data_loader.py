
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .preprocessing import encode_example, PAD

class ARTDataset(Dataset):
    """
    PyTorch dataset for ART examples.
    """

    def __init__(self, hf_dataset, vocab):
        self.examples = [
            encode_example(example, vocab)
            for example in hf_dataset
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    """
    Pad sequences dynamically to the longest
    sequence in the batch.
    """

    sequences = [
        torch.tensor(item["input_ids"], dtype=torch.long)
        for item in batch
    ]

    lengths = torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0  # [PAD] token
    )

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long
    )

    return {
        "input_ids": padded_sequences,
        "lengths": lengths,
        "labels": labels
    }

def create_dataloader(
    hf_dataset,
    vocab,
    batch_size=32,
    shuffle=False
):
    """
    Create a PyTorch DataLoader for ART.
    """

    art_dataset = ARTDataset(hf_dataset, vocab)

    return DataLoader(
        art_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
