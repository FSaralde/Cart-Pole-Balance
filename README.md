# CartPole DQN — My First Reinforcement Learning Project

This is my first reinforcement learning (RL) project: an AI agent that teaches itself to balance a pole on a moving cart, using **Deep Q-Learning (DQN)** with [PyTorch](https://pytorch.org/) and [TorchRL](https://github.com/pytorch/rl).

It starts out acting randomly and the pole falls in about **9 steps**. After about a minute of training it reaches the **maximum score of 500 out of 500**.

## The problem

[CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) is a classic RL benchmark:

- A pole is hinged on top of a cart that moves along a track.
- Every step, the agent chooses to push the cart **left** or **right**.
- It gets **+1 reward** for each step the pole stays upright.
- The episode ends if the pole tilts more than 12°, the cart leaves the track, or it reaches 500 steps (a perfect score).

The agent sees four numbers: cart position, cart velocity, pole angle and pole angular velocity. Nobody tells it how to balance the pole. It has to work that out from the reward alone, by trial and error.

## How it works

The agent uses a neural network (two hidden layers of 256 units) that predicts how much future reward each action will lead to (its **Q-value**). It picks the action with the higher Q-value.

Training repeats these steps:

1. **Explore.** It starts with 100% random actions and gradually lowers that to 4%, so it tries new things early and uses what it has learned later.
2. **Remember.** Each step is stored in a **replay buffer** of past experience.
3. **Learn.** It samples random batches from the buffer and updates the network so its predictions better match the reward it actually gets.
4. **Stabilize.** A separate **target network**, copied from the main network every 10 updates, gives it a steady goal to learn toward.
5. **Evaluate.** Every ~2,000 steps it plays 10 games with no randomness, saves the best model, and stops once it averages 500.

## Results

| | Average reward |
|---|---|
| Untrained agent (`rl.py`) | ~9 / 500 |
| Trained agent (`train.py`) | **500 / 500** |

## Getting started

**Requirements:** Python 3.10–3.13. TorchRL doesn't support newer versions yet.

```bash
# Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python rl.py              # run the untrained agent once (baseline)
python train.py           # train the agent; saves cartpole_dqn.pt
python train.py --watch   # open a window and watch the trained agent play
```

Press **Ctrl+C** in the terminal to stop at any time.

## Project files

| File | What it does |
|---|---|
| `rl.py` | Runs one episode with an untrained network to show the baseline |
| `train.py` | Trains the DQN agent, saves the best model, and has a `--watch` mode |
| `cartpole_dqn.pt` | The saved trained model (created by `train.py`) |
| `requirements.txt` | Python dependencies |

## What I learned

- **Reward is the only teacher.** The agent finds a balancing strategy without ever being shown one.
- **DQN can be unstable.** My first version reached ~380 and then "forgot" what it had learned, dropping back to ~150. Training longer didn't fix it.
- **Hyperparameters matter a lot.** The fixes were a bigger network, copying the target network every 10 updates, more updates per batch of experience, Double DQN, and stopping as soon as it hit 500.
- **Save the best model, not just the last one.** RL performance can change a lot from one evaluation to the next.

## Next steps

- [ ] Plot the reward curve over training
- [ ] Try a harder environment like `LunarLander-v3` or `Acrobot-v1`
- [ ] Compare DQN with a policy-gradient method like PPO
- [ ] Record a video or GIF of the trained agent
