import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

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

@app.entrypoint
def main(args):

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                    enable_wind=True, wind_power=15.0, turbulence_power=1.5,
                    render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env,
        video_folder = args["export"]["video_folder"]
    )

    obs, _ = env.reset()

    obs_size = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    print(f"Observation size: {obs_size}")
    print(f"Actions: {n_actions}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defining the model
    model = nn.Sequential(
        nn.Linear(obs_size, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, n_actions)
    ).to(device)

    model.load_state_dict(torch.load(args["export"]["model"], weights_only=True))

    done = False
    sm = nn.Softmax(dim=1)

    print("Recording final evaluation episode...")
    while not done:
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_v = obs_v.to(device)
        with torch.no_grad():
            act_probs_v = sm(model(obs_v))

        action = np.argmax(act_probs_v.detach().cpu().numpy()[0])

        next_obs, _ , terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs

    print("Video saved successfully.")
    env.close()

if __name__ == "__main__":
    app.run()
