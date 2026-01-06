import numpy as np
from dataclasses import dataclass
import typing as tt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from fenn import Fenn

# ----------------------------
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Overwriting existing videos at .*",
    category=UserWarning,
)
# ----------------------------

app = Fenn()
app.disable_disclaimer()

@dataclass
class EpisodeStep:
    observation: torch.Tensor
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]

def generate_batch(env: gym.Env,
                    model: nn.Module,
                    device: str,
                    batch_size: int,
                    masked:tt.List[int]=None) -> tt.List[Episode]:
    model.eval()

    batch = []
    obs, _ = env.reset()

    episode_reward = 0.0
    episode_steps = []

    sm = nn.Softmax(dim=1)

    while len(batch) < batch_size:
        obs_v = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        obs_v = obs_v.to(device)
        with torch.no_grad():
            act_probs_v = sm(model(obs_v))

        act_probs = act_probs_v.detach().cpu().numpy()[0]

        if masked is not None:
            for command_id in masked:
                act_probs[command_id] = 0.0

            s = act_probs.sum()
            if s > 0:
                act_probs /= s
            else:
                act_probs = act_probs_v.detach().cpu().numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        episode_reward += float(reward)
        step = EpisodeStep(observation=obs_v.squeeze(0), action=action)
        episode_steps.append(step)

        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)

            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()

        obs = next_obs

    model.train()
    return batch


def filter_batch(batch, percentile):
    rewards = [ep.reward for ep in batch]
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    # determine the number of selected episodes ("elite")
    elite_count = sum(1 for ep in batch if ep.reward >= reward_bound)
    elite_frac = elite_count / max(1, len(batch))

    train_obs, train_act = [], []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        for step in episode.steps:
            train_obs.append(step.observation)
            train_act.append(step.action)

    obs_v = torch.stack(train_obs, dim=0)
    act_v = torch.as_tensor(train_act, dtype=torch.long)

    return obs_v, act_v, reward_bound, reward_mean, elite_frac

@app.entrypoint
def main(args):

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                    enable_wind=False, wind_power=15.0, turbulence_power=1.5,
                    render_mode="rgb_array")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=args["env"]["max_episode_steps"])


    obs_size = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    print(f"Observation size: {obs_size}")
    print(f"Actions: {n_actions}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------

    # Defining the model
    model = nn.Sequential(
        nn.Linear(obs_size, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, n_actions)
    ).to(device)

    model.load_state_dict(torch.load("checkpoint.pth", weights_only=True))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=float(args["train"]["lr"]))

    print(model)
    # ---------------------------------------------------------

    # TensorBoard writer
    logdir = args["export"]["tensorboard"]
    writer = SummaryWriter(log_dir=logdir)

    model.train()
    for epoch in range(1, args["sampling"]["epochs"] + 1):

        batch = generate_batch(env, model, device, args["sampling"]["batch_size"])

        # rollout stats from the raw batch
        ep_len_mean = float(np.mean([len(ep.steps) for ep in batch]))

        obs_v, acts_v, reward_b, reward_m, elite_frac = filter_batch(batch, args["sampling"]["percentile"])
        obs_v = obs_v.to(device)
        acts_v = acts_v.to(device)

        for i in range(10):
            optimizer.zero_grad()
            action_scores_v = model(obs_v)
            loss_v = loss_fn(action_scores_v, acts_v)
            loss_v.backward()

            optimizer.step()

            print(f"[{epoch}]: loss={loss_v.item():.6f}, reward_mean={reward_m:.1f}, rw_bound={reward_b:.1f}")

        # TensorBoard logging
        with torch.no_grad():
            probs = torch.softmax(action_scores_v, dim=1)
            entropy = (-(probs * torch.log(probs + 1e-8)).sum(dim=1)).mean().item()
            probs_mean = probs.mean(dim=0)  # (n_actions,)
            elite_acc = (action_scores_v.argmax(dim=1) == acts_v).float().mean().item()

        writer.add_scalar("rollout/ep_rew_mean", reward_m, epoch)
        writer.add_scalar("rollout/ep_len_mean", ep_len_mean, epoch)
        writer.add_scalar("cem/reward_bound", reward_b, epoch)
        writer.add_scalar("cem/elite_frac", elite_frac, epoch)

        writer.add_scalar("train/loss_ce", loss_v.item(), epoch)
        writer.add_scalar("train/elite_acc", elite_acc, epoch)
        writer.add_scalar("policy/entropy", entropy, epoch)
        writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], epoch)

        writer.add_scalars(
            "policy/action_prob_mean",
            {f"a{i}": probs_mean[i].item() for i in range(n_actions)},
            epoch
        )
        writer.add_scalar("train/elite_samples", int(obs_v.shape[0]), epoch)

    model.eval()
    model = model.to("cpu")
    torch.save(model.state_dict(), args["export"]["model"])

    env.close()
    writer.close()


if __name__ == "__main__":
    app.run()
