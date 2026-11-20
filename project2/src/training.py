
"""Reusable training, validation and checkpoint utilities for ART BERT."""

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    """BF16 on supported CUDA GPUs; otherwise FP32."""
    enabled = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    accumulation_steps=1,
    max_grad_norm=1.0,
    run=None,
    optimizer_step=0,
    scheduler=None,
):
    if len(loader) == 0:
        raise ValueError("Training DataLoader is empty")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    correct = 0
    count = 0
    pending_samples = 0

    for index, batch in enumerate(loader):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)
        n = targets.size(0)

        with autocast_context(device):
            logits = model(ids, mask)

            if logits.shape != (n, 2):
                raise ValueError(
                    f"Expected [{n}, 2] logits, "
                    f"got {tuple(logits.shape)}"
                )

            loss = loss_fn(logits.float(), targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {index}"
            )

        # Sample-weighted gradient accumulation.
        (loss * n).backward()
        pending_samples += n

        total_loss += loss.item() * n
        correct += (
            logits.detach().argmax(dim=1) == targets
        ).sum().item()
        count += n

        update_now = (
            (index + 1) % accumulation_steps == 0
            or (index + 1) == len(loader)
        )

        if update_now:
            # Normalize the accumulated gradients.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending_samples)

            # Returns norm BEFORE clipping.
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    "Non-finite gradient norm"
                )

            lr = optimizer.param_groups[0]["lr"]

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            pending_samples = 0
            optimizer_step += 1

            if run is not None:
                run.log({
                    "optimizer_step": optimizer_step,
                    "train/learning_rate": lr,
                    "train/grad_norm": float(grad_norm),
                })

    return {
        "loss": total_loss / count,
        "accuracy": correct / count,
    }, optimizer_step

@torch.inference_mode()
def evaluate(model, loader, device):
    """Evaluate ART; class 0 = H1, class 1 = H2."""

    if len(loader) == 0:
        raise ValueError("Evaluation DataLoader is empty")

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    confidence_sum = 0.0
    count = 0

    # Rows: true labels, columns: predicted labels
    confusion = torch.zeros(2, 2, dtype=torch.long)

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        n = targets.size(0)

        with autocast_context(device):
            logits = model(ids, mask)
            loss = loss_fn(logits.float(), targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite evaluation loss"
            )

        predictions = logits.argmax(dim=1)

        confusion += torch.bincount(
            (
                targets.cpu() * 2
                + predictions.cpu()
            ).long(),
            minlength=4,
        ).reshape(2, 2)

        probabilities = torch.softmax(
            logits.float(), dim=1
        )

        confidence_sum += (
            probabilities.max(dim=1).values.sum().item()
        )

        total_loss += loss.item() * n
        count += n

    recalls = []
    f1_scores = []

    for cls in (0, 1):
        tp = confusion[cls, cls].item()
        support = confusion[cls, :].sum().item()
        predicted = confusion[:, cls].sum().item()

        recall = tp / support if support else 0.0

        denominator = support + predicted
        f1 = 2 * tp / denominator if denominator else 0.0

        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "loss": total_loss / count,
        "accuracy": confusion.diag().sum().item() / count,
        "macro_f1": sum(f1_scores) / 2,
        "recall_h1": recalls[0],
        "recall_h2": recalls[1],
        "pred_h1_rate": confusion[:, 0].sum().item() / count,
        "mean_confidence": confidence_sum / count,
        "confusion_matrix": confusion.tolist(),
    }

def setup_wandb_metrics(run):
    if run is None:
        return

    run.define_metric("epoch")
    run.define_metric("optimizer_step")

    # Epoch-level metrics
    epoch_metrics = (
        "train/loss",
        "train/accuracy",
        "train/eval_loss",
        "train/eval_accuracy",
        "val/loss",
        "val/accuracy",
        "val/macro_f1",
        "val/recall_h1",
        "val/recall_h2",
        "val/pred_h1_rate",
        "val/mean_confidence",
        "diagnostics/accuracy_gap",
        "system/epoch_seconds",
    )

    for name in epoch_metrics:
        run.define_metric(
            name,
            step_metric="epoch",
        )

    # Optimizer-update metrics
    for name in (
        "train/learning_rate",
        "train/grad_norm",
    ):
        run.define_metric(
            name,
            step_metric="optimizer_step",
        )


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    *,
    epochs,
    checkpoint_path,
    train_eval_loader=None,
    patience=3,
    min_delta=0.0,
    accumulation_steps=1,
    max_grad_norm=1.0,
    run=None,
    scheduler=None,
    print_every=1,
):
    """Train and select the best checkpoint by validation loss."""

    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    if patience is not None and patience < 1:
        raise ValueError("patience must be positive or None")

    if min_delta < 0 or print_every < 1:
        raise ValueError("Invalid stopping/logging settings")

    device = torch.device(device)
    model.to(device)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Existing smoke-test calls remain compatible.
    # For full experiments, supply a separate training
    # evaluation loader with shuffle=False.
    if train_eval_loader is None:
        train_eval_loader = train_loader

    setup_wandb_metrics(run)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_loss = math.inf
    best_accuracy = None
    best_epoch = None

    epochs_without_improvement = 0
    optimizer_step = 0
    history = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()

        # 1. Train: weights change during this pass.
        train, optimizer_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            run=run,
            optimizer_step=optimizer_step,
            scheduler=scheduler,
        )

        # 2. Evaluate training and validation datasets
        # with the same fixed end-of-epoch model.
        train_eval = evaluate(
            model,
            train_eval_loader,
            device,
        )

        val = evaluate(
            model,
            val_loader,
            device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        # 3. Save the best validation-loss checkpoint.
        improved = val["loss"] < best_loss - min_delta

        if improved:
            best_loss = val["loss"]
            best_accuracy = val["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_loss,
                "val_accuracy": best_accuracy,
            }, checkpoint_path)

            if run is not None:
                run.summary["best/val_loss"] = best_loss
                run.summary["best/val_accuracy"] = best_accuracy
                run.summary["best/epoch"] = best_epoch

        else:
            epochs_without_improvement += 1

        # 4. Calculate all epoch-level metrics.
        metrics = {
            "epoch": epoch,

            # Online training metrics
            "train/loss": train["loss"],
            "train/accuracy": train["accuracy"],

            # Evaluation-mode training metrics
            "train/eval_loss": train_eval["loss"],
            "train/eval_accuracy": train_eval["accuracy"],

            # Validation metrics
            "val/loss": val["loss"],
            "val/accuracy": val["accuracy"],
            "val/macro_f1": val["macro_f1"],
            "val/recall_h1": val["recall_h1"],
            "val/recall_h2": val["recall_h2"],
            "val/pred_h1_rate": val["pred_h1_rate"],
            "val/mean_confidence": val["mean_confidence"],

            # Generalization
            "diagnostics/accuracy_gap": (
                train_eval["accuracy"] - val["accuracy"]
            ),

            # Efficiency
            "system/epoch_seconds": elapsed,
        }

        history.append(metrics)

        if run is not None:
            run.log(metrics)

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train['loss']:.4f} "
                f"acc={train['accuracy']:.3f} | "
                f"val loss={val['loss']:.4f} "
                f"acc={val['accuracy']:.3f} | "
                f"best epoch={best_epoch}"
            )

        # 5. Early stopping
        if (
            patience is not None
            and epochs_without_improvement >= patience
        ):
            print(
                f"Early stopping: no validation-loss "
                f"improvement in {patience} epochs."
            )
            break

    # 6. Restore best model weights.
    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state["model_state_dict"])

    # 7. Run-level W&B summaries
    if run is not None:
        run.summary["train/optimizer_steps"] = optimizer_step

        if device.type == "cuda":
            run.summary["system/peak_gpu_memory_gib"] = (
                torch.cuda.max_memory_allocated(device)
                / (1024**3)
            )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_accuracy": best_accuracy,
        "optimizer_steps": optimizer_step,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
