from __future__ import annotations
from stable_baselines3 import SAC


def train_sac_agent(env, total_timesteps: int, learning_rate: float, buffer_size: int,
                     learning_starts: int, batch_size: int, gamma: float, tau: float,
                     ent_coef: str, seed: int) -> SAC:
    model = SAC(
        "MlpPolicy", env, learning_rate=learning_rate, buffer_size=buffer_size,
        learning_starts=learning_starts, batch_size=batch_size, gamma=gamma, tau=tau,
        ent_coef=ent_coef, policy_kwargs={"net_arch": [256, 256]}, seed=seed, verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model
