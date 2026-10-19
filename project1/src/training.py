import torch

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }

def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass only
            logits = model(input_ids)
            loss = criterion(logits, labels)

            # Track metrics
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": correct / total
    }
