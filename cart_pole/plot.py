#!/usr/bin/env python3
# plot.py  (call: python plot.py)
# Reads "mcts.log" and saves "plot.png"

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LOGFILE = Path("mcts.log")

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

STEP_RE = re.compile(
    rf"""
    \[
        (?P<step>\d+)
    \]:
    \s*reward={FLOAT},
    \s*ep_return=(?P<ep_return>{FLOAT}),
    \s*root_visits=\d+,
    \s*root_mean=(?P<root_mean>{FLOAT}),
    \s*best_q=(?P<best_q>{FLOAT})
    """,
    re.VERBOSE,
)

EP_END_RE = re.compile(
    rf"\[episode_end\]:\s*ep_return=(?P<ret>{FLOAT}),\s*ep_len=(?P<len>\d+)"
)


def rolling_mean(x, window=9):
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, np.ones(window) / window, mode="valid")


def parse(text):
    steps, ep_returns, root_mean, best_q = [], [], [], []

    for m in STEP_RE.finditer(text):
        steps.append(int(m.group("step")))
        ep_returns.append(float(m.group("ep_return")))
        root_mean.append(float(m.group("root_mean")))
        best_q.append(float(m.group("best_q")))

    if not steps:
        raise RuntimeError("No matching step lines found in mcts.log")

    end = EP_END_RE.search(text)
    ep_len = int(end.group("len")) if end else None
    ep_ret = float(end.group("ret")) if end else None

    return (
        np.array(steps),
        np.array(ep_returns),
        np.array(root_mean),
        np.array(best_q),
        ep_len,
        ep_ret,
    )


def main():
    if not LOGFILE.exists():
        raise FileNotFoundError("mcts.log not found")

    text = LOGFILE.read_text(encoding="utf-8", errors="replace")
    steps, ep_returns, root_mean, best_q, ep_len, ep_ret = parse(text)

    # Style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    root_mean_s = rolling_mean(root_mean, 9)
    best_q_s = rolling_mean(best_q, 9)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    # --- Episode return ---
    axes[0].plot(steps, ep_returns, linewidth=2)
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("Rollout performance")

    # --- MCTS values ---
    axes[1].plot(steps, root_mean_s, linewidth=2.2, label="Root mean value")
    axes[1].plot(steps, best_q_s, linewidth=2.2, label="Chosen child value")
    axes[1].set_ylabel("Estimated value")
    axes[1].set_xlabel("Environment step")
    axes[1].set_title("MCTS estimated value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("plot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()