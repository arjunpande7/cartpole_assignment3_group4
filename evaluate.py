"""
Evaluate trained networks with the official test procedure of test_script.py,
and record extra measurements for the analysis.

The episode logic is copied exactly from test_pole_length() / test_script():
30 pole lengths (np.linspace(0.4, 1.8, 30)), 10 episodes per length, a fresh
CartPole-v1 per episode, greedy actions, and the same force schedule.
The only differences:
  - episode i (0-9) is reset with seed 1000 + i, so every config faces the
    same start states;
  - we also record the phase in which the episode ended and the mean |theta|
    and |x| over the first 500 steps (the default-force phase);
  - an optional --max_steps cap, for quick checks only.

Usage (from the repo root):
  python evaluate.py --weights weights/baseline_seed0.pth --config baseline --seed 0 --out results/baseline_seed0_eval.csv
  python evaluate.py --all
"""
import argparse
import os

import gym
import numpy as np
import pandas as pd
import torch

from src.network import QNetwork

POLE_LENGTHS = np.linspace(0.4, 1.8, 30)   # same as test_script.py
EPISODES_PER_LENGTH = 10                    # same as test_script.py
RESET_SEED_START = 1000                     # episode i uses reset seed 1000 + i
NORMAL_PHASE_END = 500                      # first force change happens at step 500
WIND_PHASE_END = 1000                       # increasing force starts after step 1000


def phase_of_failure(steps):
    """Return the test phase in which an episode of 'steps' steps ended."""
    if steps <= NORMAL_PHASE_END:
        return "normal"
    if steps <= WIND_PHASE_END:
        return "wind"
    return "increasing"


def run_episode(env, q_network, reset_seed, max_steps=None):
    """
    Run one test episode, copied from test_pole_length() in test_script.py.

    Returns:
        steps: the total reward (= number of steps survived).
        mean_abs_theta, mean_abs_x: means of |theta| and |x| over the first
            min(steps, 500) steps (the state after each step).
    """
    wind = 25
    state = env.reset(seed=reset_seed)[0]          # only difference: a fixed reset seed
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    abs_thetas = []
    abs_xs = []

    while not done:
        action = q_network(state).argmax().item()
        next_state, reward, done, _, __ = env.step(action)

        # extra measurement: stability during the default-force phase
        if total_reward < NORMAL_PHASE_END:
            abs_xs.append(abs(next_state[0]))
            abs_thetas.append(abs(next_state[2]))

        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        state = next_state
        total_reward += reward

        # force schedule, exactly as in test_script.py
        if total_reward >= 500 and total_reward <= 1000:
            if total_reward % wind == 0:
                env.unwrapped.force_mag = 75

        if total_reward > 1000:
            env.unwrapped.force_mag = 25 + (0.01 * total_reward)

        # optional cap for quick checks (not used for the real results)
        if max_steps is not None and total_reward >= max_steps:
            break

    return int(total_reward), float(np.mean(abs_thetas)), float(np.mean(abs_xs))


def evaluate_weights(weights_path, config_name, seed, max_steps=None):
    """
    Evaluate one weight file on all 30 pole lengths x 10 episodes.

    Returns:
        A pandas DataFrame with one row per episode.
    """
    q_network = QNetwork(4, 2)
    q_network.load_state_dict(torch.load(weights_path, weights_only=True))
    q_network.eval()

    rows = []
    with torch.no_grad():                      # no gradients needed for testing
        for length in POLE_LENGTHS:
            for episode in range(EPISODES_PER_LENGTH):
                env = gym.make("CartPole-v1")
                env.unwrapped.length = length
                steps, mean_abs_theta, mean_abs_x = run_episode(
                    env, q_network, RESET_SEED_START + episode, max_steps)
                env.close()
                rows.append({
                    "config": config_name,
                    "seed": seed,
                    "pole_length": round(float(length), 4),
                    "episode": episode,
                    "steps": steps,
                    "phase_of_failure": phase_of_failure(steps),
                    "mean_abs_theta": mean_abs_theta,
                    "mean_abs_x": mean_abs_x,
                })
            last = rows[-EPISODES_PER_LENGTH:]
            print(f"  length {length:.2f}: mean steps = {np.mean([r['steps'] for r in last]):.1f}")
    return pd.DataFrame(rows)


def find_weight_files(weights_dir):
    """Return (path, config, seed) for every '<config>_seed<N>.pth' file in weights_dir."""
    found = []
    for name in sorted(os.listdir(weights_dir)):
        if name.endswith(".pth") and "_seed" in name:
            config_name, seed_text = name[:-len(".pth")].rsplit("_seed", 1)
            if seed_text.isdigit():
                found.append((os.path.join(weights_dir, name), config_name, int(seed_text)))
    return found


def main():
    """Read the command-line arguments and evaluate one or all weight files."""
    parser = argparse.ArgumentParser(description="Evaluate trained networks with the official test procedure.")
    parser.add_argument("--weights", help="path to one .pth weight file")
    parser.add_argument("--config", help="config name, e.g. baseline")
    parser.add_argument("--seed", type=int, help="training seed of this weight file")
    parser.add_argument("--out", help="output CSV path")
    parser.add_argument("--max_steps", type=int, default=None, help="optional step cap, for quick checks only")
    parser.add_argument("--all", action="store_true", help="evaluate every <config>_seed<N>.pth in --weights_dir")
    parser.add_argument("--weights_dir", default="weights")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    if args.all:
        jobs = []
        for path, config_name, seed in find_weight_files(args.weights_dir):
            out = os.path.join(args.results_dir, f"{config_name}_seed{seed}_eval.csv")
            jobs.append((path, config_name, seed, out))
    else:
        if args.weights is None or args.config is None or args.seed is None or args.out is None:
            parser.error("give --weights, --config, --seed and --out, or use --all")
        jobs = [(args.weights, args.config, args.seed, args.out)]

    for path, config_name, seed, out in jobs:
        print(f"Evaluating {path}")
        results = evaluate_weights(path, config_name, seed, args.max_steps)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        results.to_csv(out, index=False)
        print(f"Saved {out}: overall mean steps = {results['steps'].mean():.1f}\n")


if __name__ == "__main__":
    main()
