
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


CHECKPOINT = "google-bert/bert-base-uncased"


class ARTPairwiseBERT(nn.Module):
    """
    Candidate-wise BERT scoring for ART.

    Input:
        input_ids:     [B, 2, L]
        attention_mask:[B, 2, L]

    Output:
        logits:        [B, 2]
    """

    def __init__(self, initialization="random"):
        super().__init__()

        if initialization not in ("random", "pretrained"):
            raise ValueError(
                "initialization must be 'random' or 'pretrained'"
            )

        config = BertConfig.from_pretrained(CHECKPOINT)

        if initialization == "random":
            self.encoder = BertModel(
                config,
                add_pooling_layer=False,
            )
        else:
            self.encoder = BertModel.from_pretrained(
                CHECKPOINT,
                add_pooling_layer=False,
            )

        self.classifier = nn.Linear(
            config.hidden_size,
            1,
        )

        self.initialization = initialization

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 3:
            raise ValueError("Expected input shape [B, 2, L]")

        batch_size, num_choices, seq_len = input_ids.shape

        if num_choices != 2:
            raise ValueError("ART requires two hypotheses")

        if attention_mask.shape != input_ids.shape:
            raise ValueError("Attention mask shape mismatch")

        # [B, 2, L] -> [B*2, L]
        flat_ids = input_ids.reshape(-1, seq_len)
        flat_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )

        # Final-layer CLS representation: [B*2, 768]
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # One score per candidate: [B*2, 1]
        scores = self.classifier(cls_embeddings)

        # Two candidate logits per example: [B, 2]
        logits = scores.reshape(batch_size, 2)

        return logits


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from data_loader import ARTDataset, ARTCollator

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    dataset = ARTDataset("train", tokenizer, max_length=128)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=ARTCollator(tokenizer.pad_token_id),
    )

    batch = next(iter(loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Test the randomly initialized variant first.
    model = ARTPairwiseBERT("random").to(device)
    model.train()

    logits = model(input_ids, attention_mask)
    loss = nn.CrossEntropyLoss()(logits, labels)

    assert logits.shape == (2, 2)
    assert torch.isfinite(loss)

    loss.backward()

    assert model.classifier.weight.grad is not None
    assert any(
        p.grad is not None
        for p in model.encoder.parameters()
    )

    print("Device:", device)
    print("Input shape:", tuple(input_ids.shape))
    print("Output shape:", tuple(logits.shape))
    print("Logits:", logits.detach().cpu())
    print("Loss:", loss.item())
    print("Backward pass: OK")

    # Test hypothesis-order symmetry in evaluation mode.
    model.eval()

    with torch.inference_mode():
        original = model(input_ids, attention_mask)

        swapped = model(
            input_ids.flip(dims=[1]),
            attention_mask.flip(dims=[1]),
        )

        assert torch.allclose(
            swapped,
            original.flip(dims=[1]),
            atol=1e-5,
            rtol=1e-4,
        )

    print("Hypothesis-swap check: OK")
    print("All model checks passed!")
