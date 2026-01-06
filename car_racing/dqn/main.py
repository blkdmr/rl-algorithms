#!/usr/bin/env python3
# dqn_carracing_v3_processobs_only.py

import os
import random
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    env_id: str = "CarRacing-v3"
    seed: int = 0

    total_steps: int = 300_000
    buffer_size: int = 200_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-4

    learn_starts: int = 20_000
    train_every: int = 4
    target_update_every: int = 5_000

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000  # exp decay

    ckpt_path: str = "checkpoints/dqn_carracing_v3.pt"
    video_folder: str = "videos"


cfg = Config()


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int) -> float:
    return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-step / cfg.eps_decay_steps)


def make_env(render_mode=None) -> gym.Env:
    # discrete actions for DQN: continuous=False => Discrete(5) [web:5]
    env = gym.make(cfg.env_id, continuous=False, render_mode=render_mode)
    return env


def process_obs(obs: np.ndarray) -> torch.Tensor:
    """
    Converts CarRacing obs (96,96,3) uint8 -> (1,3,96,96) float32 in [0,1]. [web:5]
    """
    x = torch.from_numpy(obs).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


# ----------------------------
# DQN Model
# ----------------------------
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        # Input: (B,3,96,96) from process_obs
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            n_flat = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------
# Save / Load
# ----------------------------
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return ckpt


# ----------------------------
# Train
# ----------------------------
def train():
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(render_mode=None)
    obs, _ = env.reset(seed=cfg.seed)

    n_actions = env.action_space.n  # Discrete(5) in this mode [web:5]
    q = DQN(n_actions).to(device)
    q_tgt = DQN(n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())

    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(cfg.buffer_size)

    ep_return, ep_len = 0.0, 0

    for step in range(1, cfg.total_steps + 1):
        eps = epsilon_by_step(step)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_v = process_obs(obs).to(device)
                qvals = q(obs_v)
                action = int(torch.argmax(qvals, dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rb.add(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_return += reward
        ep_len += 1

        if done:
            if step % 10_000 < 1_000:
                print(f"step={step} eps={eps:.3f} return={ep_return:.1f} len={ep_len}")
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # Learn
        if len(rb) >= cfg.learn_starts and step % cfg.train_every == 0:
            s, a, r, s2, d = rb.sample(cfg.batch_size)

            # Convert batch using the same process as process_obs (vectorized)
            s_t = torch.from_numpy(s).to(torch.float32) / 255.0      # (B,96,96,3)
            s2_t = torch.from_numpy(s2).to(torch.float32) / 255.0
            s_t = s_t.permute(0, 3, 1, 2).contiguous().to(device)    # (B,3,96,96)
            s2_t = s2_t.permute(0, 3, 1, 2).contiguous().to(device)

            a_t = torch.from_numpy(a).long().to(device)
            r_t = torch.from_numpy(r).float().to(device)
            d_t = torch.from_numpy(d.astype(np.float32)).float().to(device)

            q_sa = q(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = q_tgt(s2_t).max(dim=1).values
                target = r_t + cfg.gamma * (1.0 - d_t) * q_next

            loss = F.smooth_l1_loss(q_sa, target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if step % cfg.target_update_every == 0:
            q_tgt.load_state_dict(q.state_dict())

        if step % 50_000 == 0:
            save_checkpoint(cfg.ckpt_path, q, opt, step)
            print(f"Saved checkpoint: {cfg.ckpt_path} (step {step})")

    save_checkpoint(cfg.ckpt_path, q, opt, cfg.total_steps)
    print(f"Training done. Final checkpoint saved to {cfg.ckpt_path}")
    env.close()


# ----------------------------
# Test + record video
# ----------------------------
def test_model(model: nn.Module, device: str, video_folder: str):
    os.makedirs(video_folder, exist_ok=True)

    env: gym.Env = gym.make(
        cfg.env_id,
        render_mode="rgb_array",
        continuous=False,  # Discrete(5) [web:5]
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="dqn-eval",
        episode_trigger=lambda ep: True,
    )  # needs rgb_array render_mode [web:40]

    obs, _ = env.reset()
    done = False

    print("Recording final evaluation episode...")
    while not done:
        obs_v = process_obs(obs).to(device)
        with torch.no_grad():
            q_values = model(obs_v)
            action = int(torch.argmax(q_values, dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    print("Video saved successfully.")
    env.close()


def main():
    train()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Build model with correct action count (Discrete(5)) [web:5]
    tmp_env = make_env(render_mode=None)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    model = DQN(n_actions)
    load_checkpoint(cfg.ckpt_path, model, device)
    test_model(model, device, cfg.video_folder)


if __name__ == "__main__":
    main()
