"""SAC training with validation-based best-checkpoint selection."""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.monitor import Monitor

from market_maker.utils.logging import get_logger
from market_maker.utils.seeds import set_global_seed

logger = get_logger(__name__)

CURVE_COLUMNS = ["timesteps", "train_reward_mean", "val_reward_mean", "val_pnl_mean", "val_map",
                 "ent_coef", "critic_loss", "actor_loss", "is_best"]


class BestCheckpointTracker:
    """Keeps the best validation score (higher is better). Ties do not replace the incumbent."""

    def __init__(self):
        self.best_score = -np.inf
        self.best_step = None

    def update(self, score: float, step: int) -> bool:
        if np.isfinite(score) and score > self.best_score:
            self.best_score, self.best_step = float(score), int(step)
            return True
        return False


def evaluate_on_windows(model, env, starts: list[int], max_inventory: float | None = None) -> dict:
    """Deterministic rollouts of `model` on fixed start windows. Returns mean episode reward / PnL / MAP."""
    rewards, pnls, maps = [], [], []
    for i, s in enumerate(starts):
        obs, _ = env.reset(seed=1000 + i, options={"start_index": int(s)})
        done, ep_r, ep_pnl, inv = False, 0.0, 0.0, []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            ep_pnl += info["reward_components"]["pnl"]
            inv.append(abs(info["inventory"]))
            done = term or trunc
        rewards.append(ep_r)
        pnls.append(ep_pnl)
        maps.append(float(np.mean(inv)) if inv else 0.0)
    return {"val_reward_mean": float(np.mean(rewards)), "val_pnl_mean": float(np.mean(pnls)),
            "val_map": float(np.mean(maps))}


class _CaptureWriter(KVWriter):
    """SB3 logger output that remembers the most recent dumped key/values (train/ent_coef, losses)."""

    def __init__(self):
        self.latest: dict = {}

    def write(self, key_values, key_excluded, step=0):
        self.latest.update(key_values)

    def close(self):
        pass


class ValidationCheckpointCallback(BaseCallback):
    """Every `eval_freq` steps (and once more at the end) evaluate on the VALIDATION windows via `eval_fn(model)`
    -> dict(val_reward_mean, val_pnl_mean, val_map). Saves `best_model.zip` when val_reward_mean improves,
    appends a row to the training curve. `eval_fn` is injectable so the selection logic is unit-testable."""

    def __init__(self, eval_fn, eval_freq: int, out_dir: str, capture: _CaptureWriter | None = None,
                 checkpoint_freq: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.eval_fn, self.eval_freq, self.out_dir = eval_fn, max(1, int(eval_freq)), out_dir
        self.capture, self.checkpoint_freq = capture, int(checkpoint_freq)
        self.tracker = BestCheckpointTracker()
        self.rows: list[dict] = []
        self._last_eval = -1
        os.makedirs(out_dir, exist_ok=True)

    def _train_reward_mean(self):
        buf = getattr(self.model, "ep_info_buffer", None)
        return float(np.mean([e["r"] for e in buf])) if buf else float("nan")

    def _evaluate(self):
        step = int(self.num_timesteps)
        res = self.eval_fn(self.model)
        improved = self.tracker.update(res["val_reward_mean"], step)
        if improved:
            self.model.save(os.path.join(self.out_dir, "best_model"))
        lg = self.capture.latest if self.capture else {}
        self.rows.append({"timesteps": step, "train_reward_mean": self._train_reward_mean(),
                          "val_reward_mean": res["val_reward_mean"], "val_pnl_mean": res.get("val_pnl_mean", np.nan),
                          "val_map": res.get("val_map", np.nan),
                          "ent_coef": lg.get("train/ent_coef", np.nan), "critic_loss": lg.get("train/critic_loss", np.nan),
                          "actor_loss": lg.get("train/actor_loss", np.nan), "is_best": int(improved)})
        self._last_eval = step
        logger.info("step %d: val_reward %.4f (best %.4f @ %s)", step, res["val_reward_mean"],
                    self.tracker.best_score, self.tracker.best_step)

    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        if step % self.eval_freq == 0:
            self._evaluate()
        if self.checkpoint_freq and step % self.checkpoint_freq == 0:
            self.model.save(os.path.join(self.out_dir, f"checkpoint_{step}"))
        return True

    def _on_training_end(self) -> None:
        if self._last_eval != int(self.num_timesteps):
            self._evaluate()
        self.write_curve()

    def write_curve(self) -> str:
        path = os.path.join(self.out_dir, "training_curve.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CURVE_COLUMNS)
            w.writeheader()
            w.writerows(self.rows)
        return path


@dataclass
class SacTrainResult:
    best_model_path: str
    last_model_path: str
    curve_path: str
    best_step: int | None
    best_val_reward: float
    curve: list = field(default_factory=list)


def train_sac_agent(train_env, val_env, sac_cfg, *, seed: int, out_dir: str, val_starts: list[int],
                    total_timesteps: int | None = None, eval_freq: int | None = None,
                    max_inventory: float | None = None) -> SacTrainResult:
    """Train SAC on `train_env` (Monitor-wrapped here); every `eval_freq` steps score the policy on fixed
    validation windows and keep `best_model.zip`; `last_model.zip` is saved at the end. All hyperparameters come
    from `sac_cfg` (config.SacConfig). Fully seeded (global seeds, SAC seed, seeded env start sampler)."""
    set_global_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    total = int(total_timesteps if total_timesteps is not None else sac_cfg.total_timesteps)
    model = SAC(
        "MlpPolicy", Monitor(train_env), learning_rate=sac_cfg.learning_rate, buffer_size=sac_cfg.buffer_size,
        learning_starts=sac_cfg.learning_starts, batch_size=sac_cfg.batch_size, gamma=sac_cfg.gamma, tau=sac_cfg.tau,
        ent_coef=sac_cfg.ent_coef, policy_kwargs={"net_arch": list(sac_cfg.net_arch)}, seed=seed, verbose=0,
        device=sac_cfg.device,
    )
    capture = _CaptureWriter()
    model.set_logger(Logger(folder=None, output_formats=[capture]))
    cb = ValidationCheckpointCallback(
        lambda m: evaluate_on_windows(m, val_env, val_starts, max_inventory), eval_freq or sac_cfg.eval_freq,
        out_dir, capture=capture, checkpoint_freq=sac_cfg.checkpoint_freq)
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    last = os.path.join(out_dir, "last_model.zip")
    model.save(last)
    if not os.path.exists(os.path.join(out_dir, "best_model.zip")):  # no finite validation score ever: fall back
        model.save(os.path.join(out_dir, "best_model"))
    return SacTrainResult(best_model_path=os.path.join(out_dir, "best_model.zip"), last_model_path=last,
                          curve_path=os.path.join(out_dir, "training_curve.csv"), best_step=cb.tracker.best_step,
                          best_val_reward=cb.tracker.best_score, curve=cb.rows)
