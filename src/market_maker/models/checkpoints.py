from __future__ import annotations
import os
import torch


def save_checkpoint(model, path: str, extra: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra}, path)


def load_checkpoint(model_cls, path: str, **model_kwargs):
    # `extra` may contain arbitrary Python objects (not just tensors/primitives), so
    # weights_only=True (torch's safer default) would reject this payload. This is a
    # deliberate, explicit choice, not an oversight — checkpoints are only ever produced
    # by our own save_checkpoint() from trusted, locally-generated training runs.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = model_cls(**model_kwargs)
    model.load_state_dict(payload["state_dict"])
    return model, payload["extra"]
