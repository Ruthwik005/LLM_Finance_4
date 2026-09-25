from __future__ import annotations
import numpy as np
import torch
from torch import nn


def train_transformer(model, X_train, y_train_by_horizon, X_val, y_val_by_horizon,
                       epochs: int, batch_size: int, lr: float) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = {h: torch.tensor(y, dtype=torch.long) for h, y in y_train_by_horizon.items()}
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = {h: torch.tensor(y, dtype=torch.long) for h, y in y_val_by_horizon.items()}

    n = len(X_train_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train_t[idx]
            optimizer.zero_grad()
            _, logits_by_horizon = model(batch_x)
            loss = sum(criterion(logits_by_horizon[h], y_train_t[h][idx]) for h in logits_by_horizon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            _, val_logits = model(X_val_t)
            val_loss = sum(criterion(val_logits[h], y_val_t[h]) for h in val_logits)
        history["val_loss"].append(val_loss.item())

    return history
