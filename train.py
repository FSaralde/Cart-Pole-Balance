"""Train a DQN agent to balance the pole in CartPole-v1.

Run:  .venv/bin/python train.py
Watch the trained agent afterwards:  .venv/bin/python train.py --watch
"""
import sys

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

torch.manual_seed(0)
MODEL_PATH = "cartpole_dqn.pt"


def make_env(render=False):
    kwargs = {"render_mode": "human"} if render else {}
    return TransformedEnv(GymEnv("CartPole-v1", **kwargs), StepCounter())


def make_actor(env):
    return QValueActor(
        MLP(
            in_features=env.observation_spec["observation"].shape[-1],
            out_features=env.action_spec.shape[-1],
            num_cells=[256, 256],
        ),
        in_keys=["observation"],
        spec=env.action_spec,
    )


def evaluate(env, actor, episodes=5):
    """Average reward with no random exploration (500 is a perfect score)."""
    total = 0.0
    with torch.no_grad():
        for _ in range(episodes):
            r = env.rollout(max_steps=500, policy=actor)
            total += r["next", "reward"].sum().item()
    return total / episodes


def train():
    env = make_env()
    actor = make_actor(env)

    # Exploration: start fully random, decay to 4% random actions.
    explore = EGreedyModule(
        env.action_spec, eps_init=1.0, eps_end=0.04, annealing_num_steps=8_000
    )
    policy = TensorDictSequential(actor, explore)

    collector = Collector(
        env, policy, frames_per_batch=256, total_frames=102_400, init_random_frames=1_024
    )
    buffer = ReplayBuffer(storage=LazyTensorStorage(100_000), batch_size=64)

    loss_fn = DQNLoss(actor, action_space=env.action_spec, delay_value=True, double_dqn=True)
    loss_fn.make_value_estimator(gamma=0.98)
    # Copy the learning network into the "target" network every 10 updates.
    target_updater = HardUpdate(loss_fn, value_network_update_interval=10)
    optim = torch.optim.Adam(loss_fn.parameters(), lr=2.3e-3)

    best = 0.0
    for i, data in enumerate(collector):
        buffer.extend(data.reshape(-1))
        if len(buffer) >= 1_000:
            for _ in range(128):
                loss = loss_fn(buffer.sample())["loss"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optim.step()
                target_updater.step()
        explore.step(data.numel())

        if i % 8 == 0:
            score = evaluate(env, actor, episodes=10)
            print(
                f"frames {collector._frames:6d} | eps {explore.eps.item():.2f} "
                f"| eval reward {score:5.1f} / 500"
            )
            if score >= best:
                best = score
                torch.save(actor.state_dict(), MODEL_PATH)
            if score >= 500:
                print("Solved! Pole balanced for the full 500 steps.")
                break

    collector.shutdown()
    print(f"Best eval reward {best:.0f}. Saved model to {MODEL_PATH}")


def watch():
    env = make_env(render=True)
    actor = make_actor(env)
    actor.load_state_dict(torch.load(MODEL_PATH))
    print(f"Reward: {evaluate(env, actor, episodes=1):.0f} / 500")
    env.close()


if __name__ == "__main__":
    watch() if "--watch" in sys.argv else train()
