from __future__ import annotations
import os
import torch


def save_checkpoint(model, path: str, extra: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra}, path)


def load_checkpoint(model_cls, path: str, **model_kwargs):
    payload = torch.load(path, map_location="cpu")
    model = model_cls(**model_kwargs)
    model.load_state_dict(payload["state_dict"])
    return model, payload["extra"]
