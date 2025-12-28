import gymnasium as gym
import numpy as np
import math
from fenn import FENN

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
app.set_config_file("mcts.yaml")


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = np.array(state, dtype=np.float64)
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_actions = [0, 1]

    def select_child(self):
        # UCB1
        log_total_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda c: (c.wins / c.visits) + 1.41 * math.sqrt(log_total_visits / c.visits),
        )

    def expand(self, sim_env):
        action = self.untried_actions.pop()
        obs, reward, terminated, truncated, _ = sim_env.step(action)
        child_node = MCTSNode(obs, parent=self, action=action)
        self.children.append(child_node)
        return child_node, terminated, truncated


def rollout(sim_env):
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated) and total_reward < 30:
        _, reward, terminated, truncated, _ = sim_env.step(sim_env.action_space.sample())
        total_reward += float(reward)
    return total_reward


def path_actions(node):
    """Return the action sequence from root -> node (excluding root)."""
    actions = []
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    actions.reverse()
    return actions


def sync_env_to_node(sim_env, root_state, node):
    """
    Minimal, robust sync:
    - reset env
    - force CartPole internal state to root_state
    - replay actions along the tree path to reach `node`
    """
    sim_env.reset()
    # CartPole keeps its internal continuous state in env.unwrapped.state [page:1]
    sim_env.unwrapped.state = np.array(root_state, dtype=np.float64)

    terminated = False
    truncated = False
    for a in path_actions(node):
        _, _, terminated, truncated, _ = sim_env.step(a)
        if terminated or truncated:
            break
    return terminated, truncated


def mcts_search(current_obs, iterations=50):
    sim_env = gym.make("CartPole-v1", render_mode="rgb_array")
    sim_env = gym.wrappers.TimeLimit(sim_env, max_episode_steps=2000)

    root = MCTSNode(current_obs)

    for _ in range(iterations):
        node = root

        # 0) Always sync sim_env to the *current* node by replaying actions
        terminated, truncated = sync_env_to_node(sim_env, root.state, node)

        # 1) Selection
        while (not terminated) and (not truncated) and (not node.untried_actions) and node.children:
            node = node.select_child()
            terminated, truncated = sync_env_to_node(sim_env, root.state, node)

        # 2) Expansion
        if (not terminated) and (not truncated) and node.untried_actions:
            node, terminated, truncated = node.expand(sim_env)

        # 3) Simulation
        reward = 0.0 if (terminated or truncated) else rollout(sim_env)

        # 4) Backprop
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

    sim_env.close()

    # If root never expanded (can happen with very low iterations), default to random
    if not root.children:
        return 0

    return max(root.children, key=lambda c: c.visits).action


@app.entrypoint
def main(args):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="video")

    obs, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = mcts_search(obs, iterations=40)
        obs, reward, terminated, truncated, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    app.run()
