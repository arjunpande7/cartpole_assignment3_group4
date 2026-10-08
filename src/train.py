import argparse
import os
import random
import time
import gym
import numpy as np
import pandas as pd
import torch
import yaml
from src.agent import DQNAgent
from src.length_sampler import make_sampler
from src.rewards import compute_reward


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_epsilon(total_steps, config):
    fraction = min(total_steps / config["epsilon_decay_steps"], 1.0)
    return config["epsilon_start"] + fraction * (config["epsilon_end"] - config["epsilon_start"])


def train(config, seed, out_dir, weights_dir):

    set_seed(seed)
    rng = np.random.default_rng(seed)
    run_name = f"{config['name']}_seed{seed}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    env = gym.make('CartPole-v1').unwrapped
    agent = DQNAgent(config)
    sampler = make_sampler(config, rng)

    episode_log = []
    sampler_log = []
    episode_lengths = []
    total_steps = 0
    episode = 0
    start_time = time.time()

    while total_steps < config["total_steps"]:

        length = sampler.sample()
        env.length = length

        if episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        steps = 0
        shaped_return = 0
        losses = []

        while True:
            epsilon = get_epsilon(total_steps, config)
            action = agent.act(state, epsilon)
            next_state, _, terminated, truncated, info = env.step(action)

            reward = compute_reward(next_state, config)
            agent.buffer.push(state, action, reward, next_state, terminated)
            state = next_state

            steps += 1
            total_steps += 1
            shaped_return += reward

            if len(agent.buffer) >= config["learning_starts"]:
                losses.append(agent.learn())

            if total_steps % config["target_update"] == 0:
                agent.update_target()

            if total_steps % 10000 == 0:
                sampler_log.append([total_steps] + list(sampler.get_distribution()))

            if total_steps % 50000 == 0: #This is just as a safety save if colab disconnects
                agent.save(os.path.join(run_dir, "checkpoint.pth"))

            if terminated or steps >= config["episode_cap"] or total_steps >= config["total_steps"]:
                break

        sampler.update(length, steps)

        if len(losses) > 0:
            mean_loss = sum(losses) / len(losses)
        else:
            mean_loss = None

        episode_log.append({
            "total_steps": total_steps,
            "episode": episode,
            "pole_length": length,
            "episode_length": steps,
            "shaped_return": shaped_return,
            "epsilon": epsilon,
            "mean_loss": mean_loss,
        })
        episode_lengths.append(steps)
        episode += 1

        if episode % 25 == 0:
            average = sum(episode_lengths[-25:]) / 25
            minutes = (time.time() - start_time) / 60
            print(f"Episode {episode} | steps {total_steps}/{config['total_steps']} | "
                  f"avg length last 25 = {round(average, 1)} | epsilon = {round(epsilon, 3)} | {round(minutes, 1)} min")
            pd.DataFrame(episode_log).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    env.close()

    agent.save(os.path.join(weights_dir, f"{run_name}.pth"))
    agent.save(os.path.join(run_dir, f"{run_name}.pth"))
    pd.DataFrame(episode_log).to_csv(os.path.join(run_dir, "train_log.csv"), index=False)
    bin_columns = [f"bin_{i}" for i in range(len(sampler.get_distribution()))]
    pd.DataFrame(sampler_log, columns=["total_steps"] + bin_columns).to_csv(
        os.path.join(run_dir, "sampler_log.csv"), index=False)

    print(f"Done: {run_name}, {episode} episodes, {round((time.time() - start_time) / 60, 1)} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="e.g. config/baseline.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="runs", help=" logs and checkpoints go (use Drive on Colab) here")
    parser.add_argument("--weights_dir", default="weights", help="the final weights go here")
    parser.add_argument("--total_steps", type=int, default=None, help="override the budget, only for quick tests")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.total_steps is not None:
        config["total_steps"] = args.total_steps

    train(config, args.seed, args.out_dir, args.weights_dir)