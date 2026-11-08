import torch
from torchrl.envs import GymEnv
from torchrl.modules import MLP, QValueActor

env = GymEnv("CartPole-v1")
actor = QValueActor(
    MLP(
        in_features=env.observation_spec["observation"].shape[-1],
        out_features=2,
        num_cells=[64, 64],
    ),
    in_keys=["observation"],
    spec=env.action_spec,
)
rollout = env.rollout(max_steps=200, policy=actor)
print(
    f"Collected {rollout.shape[0]} steps, total reward: {rollout['next', 'reward'].sum().item():.0f}"
)