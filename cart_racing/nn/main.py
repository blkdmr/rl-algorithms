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

@dataclass
class EpisodeStep:
    observation: torch.Tensor
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]

def process_obs(obs: np.ndarray) -> torch.Tensor:

    #transform = timm.data.create_transform(**data_config, is_training=False)

    base_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224,0.225))
        #T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    img = Image.fromarray(obs.astype(np.uint8))
    x = base_transform(img)
    x = x.to(dtype=torch.float32)
    return x.unsqueeze(0)

    #return obs
    #weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    #gray = np.dot(obs[..., :3], weights).astype(np.float32)   # (96, 96)
    #gray = gray / 255.0                                       # normalize to [0, 1]
    #return gray.flatten()                                     # (9216,)

def test_model(model: nn.Module, device:str, video_folder: str):

    env =  gym.make("CarRacing-v3",
                    render_mode="rgb_array",
                    lap_complete_percent=0.95,
                    domain_randomize=False,
                    continuous=False)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True
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

    # the final batch of episodes
    batch = []

    # reset the env and get the first observation
    obs, _ = env.reset()

    episode_reward = 0.0
    episode_steps = []

    # used to extract a list of action probabilities
    # from the nn model
    sm = nn.Softmax(dim=1)

    while len(batch) < batch_size:

        obs_v = process_obs(obs)
        obs_v = obs_v.to(device)
        # retrieve the action probabilities for the observation
        with torch.no_grad():
            act_probs_v = sm(model(obs_v))

        act_probs = act_probs_v.detach().cpu().numpy()[0]

        # mask disabled commands and renormalize
        if masked is not None:
            for command_id in masked:
                act_probs[command_id] = 0.0

            s = act_probs.sum()
            if s > 0:
                act_probs /= s
            else:
                act_probs = act_probs_v.detach().cpu().numpy()[0]

        # choose an action using the generated distribution
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        episode_reward += float(reward)
        step = EpisodeStep(observation=obs_v.squeeze(0), action=action)
        episode_steps.append(step)

        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)

            # resets everything
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

    train_obs, train_act = [], []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        for step in episode.steps:
            train_obs.append(step.observation)
            train_act.append(step.action)

    obs_v = torch.stack(train_obs, dim=0)
    act_v = torch.as_tensor(train_act, dtype=torch.long)
    return obs_v, act_v, reward_bound, reward_mean

@app.entrypoint
def main(args):

    # Loading the enviroment
    # ---------------------------
    env =  gym.make("CarRacing-v3",
                    render_mode="rgb_array",
                    lap_complete_percent=0.95,
                    domain_randomize=True,
                    continuous=False)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=700)
    n_actions = int(env.action_space.n)

    # Defining the model
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model('resnet18', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_actions)
    model = model.to(device, dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # Training loop
    # ---------------------------
    round = 0
    model.train()
    while round < args["sampling"]["rounds"]:
        round += 1

        if round < 10:
            batch = generate_batch(env, model, device, args["sampling"]["batch_size"], masked=[0,4])
        else:
            batch = generate_batch(env, model, device, args["sampling"]["batch_size"])

        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, args["sampling"]["percentile"])
        obs_v = obs_v.to(device)
        acts_v = acts_v.to(device)

        optimizer.zero_grad()
        action_scores_v = model(obs_v)
        loss_v = loss_fn(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            round, loss_v.item(), reward_m, reward_b))

    # Final operations
    # ---------------------------
    model.eval()
    torch.save(model.state_dict(), args["export"]["model"])
    test_model(model, device, args["export"]["video_folder"])
    env.close()

if __name__ == "__main__":
    app.run()