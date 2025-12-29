import numpy as np
from dataclasses import dataclass
import typing as tt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from fenn import FENN
import timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from fenn.notification import Notifier
from fenn.notification.services import Telegram

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


app = FENN()
app.set_config_file("full_ft.yaml")

notifier = Notifier()
notifier.add_service(Telegram)


@dataclass
class EpisodeStep:
    observation: torch.Tensor
    action: int


@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]


def process_obs(obs: np.ndarray, device="cpu") -> torch.Tensor:
    base_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224,0.225))
    ])

    img = Image.fromarray(obs.astype(np.uint8))
    x = base_transform(img)
    x = x.to(dtype=torch.float32)
    return x.unsqueeze(0)


def test_model(model: nn.Module, device:str, video_folder: str):
    env: gym.Env = gym.make("CarRacing-v3",
                            render_mode="rgb_array",
                            continuous=False)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder
    )

    obs, _ = env.reset()

    done = False
    sm = nn.Softmax(dim=1)

    print("Recording final evaluation episode...")
    while not done:
        obs_v = process_obs(obs)
        obs_v = obs_v.to(device)
        with torch.no_grad():
            act_probs_v = sm(model(obs_v))

        action = np.argmax(act_probs_v.detach().cpu().numpy()[0])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs

    print("Video saved successfully.")
    env.close()


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
        obs_v = process_obs(obs)
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


def _grad_norm_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2).item()
        total += param_norm ** 2
    return float(total ** 0.5)


@app.entrypoint
def main(args):

    env =  gym.make("CarRacing-v3",
                    render_mode="rgb_array",
                    continuous=False)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=args["env"]["max_episode_steps"])
    n_actions = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    model = timm.create_model('resnet18', pretrained=True)

    for param in model.parameters():
        param.requires_grad = args["model"]["backbone"]

    model.fc = nn.Linear(model.fc.in_features, n_actions)
    model = model.to(device, dtype=torch.float32)
    # ---------------------------------------------------------

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args["train"]["lr"])

    # TensorBoard writer
    logdir = args["export"]["tensorboard"]
    writer = SummaryWriter(log_dir=logdir)

    round = 0
    model.train()
    while round < args["sampling"]["epochs"]:
        round += 1

        if round < 3:
            batch = generate_batch(env, model, device, args["sampling"]["batch_size"], masked=[0,1,2,4])
        elif round < 10:
            batch = generate_batch(env, model, device, args["sampling"]["batch_size"], masked=[0,4])
        else:
            batch = generate_batch(env, model, device, args["sampling"]["batch_size"])

        # rollout stats from the raw batch
        ep_len_mean = float(np.mean([len(ep.steps) for ep in batch]))

        obs_v, acts_v, reward_b, reward_m, elite_frac = filter_batch(batch, args["sampling"]["percentile"])
        obs_v = obs_v.to(device)
        acts_v = acts_v.to(device)

        optimizer.zero_grad()
        action_scores_v = model(obs_v)
        loss_v = loss_fn(action_scores_v, acts_v)
        loss_v.backward()

        # grad norm BEFORE optimizer step
        grad_norm = _grad_norm_l2(model)

        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            round, loss_v.item(), reward_m, reward_b))

        # TensorBoard logging
        with torch.no_grad():
            probs = torch.softmax(action_scores_v, dim=1)
            entropy = (-(probs * torch.log(probs + 1e-8)).sum(dim=1)).mean().item()
            probs_mean = probs.mean(dim=0)  # (n_actions,)
            elite_acc = (action_scores_v.argmax(dim=1) == acts_v).float().mean().item()

        notifier.notify(f"Round {round} [ENDED] with loss {loss_v.item():.4f} and mean reward {reward_m:.1f}")

        writer.add_scalar("rollout/ep_rew_mean", reward_m, round)
        writer.add_scalar("rollout/ep_len_mean", ep_len_mean, round)
        writer.add_scalar("cem/reward_bound", reward_b, round)
        writer.add_scalar("cem/elite_frac", elite_frac, round)

        writer.add_scalar("train/loss_ce", loss_v.item(), round)
        writer.add_scalar("train/elite_acc", elite_acc, round)
        writer.add_scalar("policy/entropy", entropy, round)
        writer.add_scalar("optim/grad_norm", grad_norm, round)
        writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], round)

        writer.add_scalars(
            "policy/action_prob_mean",
            {f"a{i}": probs_mean[i].item() for i in range(n_actions)},
            round
        )
        writer.add_scalar("train/elite_samples", int(obs_v.shape[0]), round)

    test_model(model, device, args["export"]["video_folder"])

    model.eval()
    model = model.to("cpu")
    torch.save(model.state_dict(), args["export"]["model"])
    
    env.close()
    writer.close()


if __name__ == "__main__":
    app.run()
