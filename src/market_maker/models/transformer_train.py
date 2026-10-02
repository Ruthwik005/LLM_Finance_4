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


# ---------------------------------------------------------------------------
# Production training loop (DataLoader over lazily built windows, early stopping, checkpoint, curves).
# The legacy `train_transformer` above (in-memory tensors) is kept for tests/reference.
# ---------------------------------------------------------------------------
import copy
import csv
import os
import time

from torch.utils.data import DataLoader

from market_maker.features.temporal_dataset import IGNORE_LABEL, WindowDataset


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def class_weights_from_counts(counts: dict[int, np.ndarray]) -> dict[int, torch.Tensor]:
    """Inverse-frequency weights per horizon, normalised to mean 1 (absent classes get weight 0)."""
    out = {}
    for h, c in counts.items():
        c = np.asarray(c, dtype=np.float64)
        w = np.where(c > 0, c.sum() / (len(c) * np.maximum(c, 1)), 0.0)
        pos = w[c > 0]
        w = w / pos.mean() if len(pos) else w
        out[h] = torch.tensor(w, dtype=torch.float32)
    return out


def _multi_horizon_loss(logits_by_h, y, horizons, weights=None):
    """Sum over horizons of mean CE over labeled samples (IGNORE_LABEL entries masked). y: (B, n_h)."""
    total = 0.0
    for j, h in enumerate(horizons):
        mask = y[:, j] != IGNORE_LABEL
        if mask.any():
            w = None if weights is None else weights[h].to(y.device)
            total = total + nn.functional.cross_entropy(logits_by_h[h][mask], y[:, j][mask], weight=w)
    return total


@torch.no_grad()
def _evaluate(model, loader, horizons, device, weights):
    model.eval()
    loss_sum, n_batches = 0.0, 0
    correct = {h: 0 for h in horizons}
    count = {h: 0 for h in horizons}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x)
        loss = _multi_horizon_loss(logits, y, horizons, weights)
        loss_sum += float(loss)
        n_batches += 1
        for j, h in enumerate(horizons):
            mask = y[:, j] != IGNORE_LABEL
            correct[h] += int((logits[h].argmax(1)[mask] == y[:, j][mask]).sum())
            count[h] += int(mask.sum())
    return loss_sum / max(1, n_batches), {h: correct[h] / max(1, count[h]) for h in horizons}


def fit_transformer(model, train_ds: WindowDataset, val_ds: WindowDataset, *, epochs: int, batch_size: int,
                    lr: float, weight_decay: float, patience: int, device: str = "cpu", num_workers: int = 0,
                    use_class_weights: bool = False, seed: int = 0, checkpoint_path: str | None = None,
                    checkpoint_extra: dict | None = None, curve_path: str | None = None) -> dict:
    """Train on `train_ds`, early-stop on validation loss, keep (and optionally save) the best epoch.

    Loss = sum over horizons of CE (optionally inverse-frequency class weights computed on TRAIN labels only).
    Validation loss uses the same (weighted) criterion; validation accuracy per horizon and the majority-class
    accuracy of the validation labels are logged so improvement over the trivial baseline is visible.
    Returns history dict; the model is left holding the best-epoch weights.
    """
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"empty window dataset (train={len(train_ds)}, val={len(val_ds)}); "
                         "need more rows than sequence_length + horizon")
    horizons = list(model.horizons)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              generator=gen, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 512), shuffle=False, num_workers=num_workers)
    weights = class_weights_from_counts(train_ds.label_counts()) if use_class_weights else None
    val_counts = val_ds.label_counts()
    majority = {h: float(c.max() / max(1, c.sum())) for h, c in val_counts.items()}

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_majority_acc": majority,
               "best_epoch": -1, "stopped_early": False}
    best_loss, best_state, bad = float("inf"), None, 0
    rows = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            _, logits = model(x)
            loss = _multi_horizon_loss(logits, y, horizons, weights)
            if not torch.is_tensor(loss):
                continue
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        train_loss = tot / max(1, nb)
        val_loss, val_acc = _evaluate(model, val_loader, horizons, device, weights)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     **{f"val_acc_h{h}": val_acc[h] for h in horizons},
                     **{f"val_majority_acc_h{h}": majority[h] for h in horizons},
                     "seconds": time.time() - t0})
        if val_loss < best_loss - 1e-9:
            best_loss, bad = val_loss, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            history["best_epoch"] = epoch
            if checkpoint_path:
                from market_maker.models.checkpoints import save_checkpoint
                tmp = dict(checkpoint_extra or {})
                tmp.update({"best_epoch": epoch, "best_val_loss": best_loss,
                            "history": {k: v for k, v in history.items()}})
                payload_model = copy.deepcopy(model).cpu()
                save_checkpoint(payload_model, checkpoint_path, extra=tmp)
        else:
            bad += 1
            if bad >= patience:
                history["stopped_early"] = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if curve_path and rows:
        os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)
        with open(curve_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    history["best_val_loss"] = best_loss
    return history
