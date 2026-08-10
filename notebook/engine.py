import torch
MetricDict = dict[str,float]

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> MetricDict:
    """
        The training loop is as follows

        Load data
        Create batcher

        For each batch:
            - move_batch
            - clear gradient
            - forward
            - backward pass
            - update weight
            - accumulate loss & accuracy.
    """
    # Setting model to train mode.
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for batch_index, (batch_features_cpu, batch_labels_cpu) in enumerate(dataloader):
        # Switch to target device
        batch_features = batch_features_cpu.to(device)
        batch_labels = batch_labels_cpu.to(device)
        
        # Reset gradient
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(batch_features)
        loss = loss_fn(logits, batch_labels)

        # Backward pass
        loss.backward()

        # Update weight
        optimizer.step()

        batch_samples = batch_labels.shape[0]
        running_loss += loss.item() * batch_samples
        predictions = logits.argmax(dim=1)
        running_correct += (predictions == batch_labels).sum().item()
        running_samples += batch_samples
    
    return {
        "loss": running_loss / running_samples,
        "accuracy": float(running_correct) / running_samples
    }

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> MetricDict:
    """
        The training loop is as follows

        For each batch:
            - move_batch
            - forward
            - accumulate loss & accuracy.
    """
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    with torch.inference_mode():
        for batch_index, (batch_features_cpu, batch_labels_cpu) in enumerate(dataloader):
            # Switch to target device
            batch_features = batch_features_cpu.to(device)
            batch_labels = batch_labels_cpu.to(device)
            
            # Forward pass
            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)

            batch_samples = batch_labels.shape[0]
            running_loss += loss.item() * batch_samples
            predictions = logits.argmax(dim=1)
            running_correct += (predictions == batch_labels).sum().item()
            running_samples += batch_samples
        
    return {
        "loss": running_loss / running_samples,
        "accuracy": float(running_correct) / running_samples
    }