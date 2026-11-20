
import argparse
import gc
import statistics
import time

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


MODELS = {
    "BERT-Mini": "google/bert_uncased_L-4_H-256_A-4",
    "BERT-Base": "google-bert/bert-base-uncased",
}


class CandidateScorer(nn.Module):
    """Score two hypotheses with one shared BERT encoder."""

    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel(config)  # Random initialization
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        batch_size, choices, seq_len = input_ids.shape
        assert choices == 2

        flat_ids = input_ids.reshape(-1, seq_len)
        flat_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
        )

        cls = outputs.last_hidden_state[:, 0, :]
        scores = self.classifier(cls)

        return scores.reshape(batch_size, 2)


def benchmark(name, checkpoint, batch_size, seq_len, steps):
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    config = BertConfig.from_pretrained(checkpoint)
    model = CandidateScorer(config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Synthetic ART-shaped batch: [batch, 2, sequence_length]
    input_ids = torch.randint(
        100, config.vocab_size,
        (batch_size, 2, seq_len),
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 2, (batch_size,), device="cuda")

    use_bf16 = torch.cuda.is_bf16_supported()
    precision = "BF16 autocast" if use_bf16 else "FP32"

    def train_step():
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = model(input_ids, attention_mask)
            assert logits.shape == (batch_size, 2)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

    # Warmup initializes optimizer state and CUDA kernels.
    for _ in range(2):
        train_step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(steps):
        start = time.perf_counter()
        train_step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    parameters = sum(p.numel() for p in model.parameters())
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    mean_time = statistics.mean(times)

    print(f"\n=== {name} ===")
    print(f"Checkpoint config: {checkpoint}")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"Parameters: {parameters:,}")
    print(f"Precision: {precision}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Mean training step: {mean_time:.3f} s")
    print(f"Peak PyTorch GPU memory: {peak_gib:.2f} GiB")

    del model, optimizer, input_ids, attention_mask, labels
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Synthetic benchmark — no ART data or pretrained weights")

    for name, checkpoint in MODELS.items():
        benchmark(
            name, checkpoint,
            args.batch_size, args.seq_len, args.steps,
        )
