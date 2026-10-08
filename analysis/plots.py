"""
Figures for the paper, made from results/<config>_seed<N>_eval.csv.

Saved in the results folder:
  1. bar_<config>.png   bar plot in the style of bar_plot() in test_script.py:
                        mean steps per pole length (mean over seeds),
                        error bars = std across seeds, title = overall average.
  2. phase_of_failure.png   per config, % of episodes ending in each test phase.
  3. mean_abs_theta.png     mean |theta| per config (mean +- std across seeds).

Usage (from the repo root):  python -m analysis.plots [--results_dir results]
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from analysis.stats import CONFIGS, load_results, per_seed_metrics

PHASES = ["normal", "wind", "increasing"]
PHASE_LABELS = {"normal": "Normal (steps 1-500)",
                "wind": "Wind / force 75 (501-1000)",
                "increasing": "Increasing force (>1000)"}


def length_means_per_seed(episodes, config):
    """Return (lengths, array of shape (n_seeds, n_lengths)) with mean steps per length per seed."""
    config_episodes = episodes[episodes["config"] == config]
    lengths = sorted(config_episodes["pole_length"].unique())
    table = []
    for seed in sorted(config_episodes["seed"].unique()):
        run = config_episodes[config_episodes["seed"] == seed]
        table.append([run[run["pole_length"] == length]["steps"].mean() for length in lengths])
    return lengths, np.array(table)


def bar_plot_config(episodes, config, out_path):
    """Bar plot of one config, in the same style as bar_plot() in test_script.py."""
    lengths, table = length_means_per_seed(episodes, config)
    avg_values = table.mean(axis=0)            # mean over seeds
    std_values = table.std(axis=0)             # std across seeds
    overall_avg = np.mean(avg_values)

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(avg_values)), avg_values, yerr=std_values, capsize=5, alpha=0.7)
    plt.xticks(range(len(avg_values)), [str(round(length, 2)) for length in lengths])
    plt.xticks(rotation=45)
    plt.xlabel("Pole length")
    plt.ylabel("Episode length")
    plt.title(f"{config}: Average score over all pole lengths = {round(overall_avg, 0)}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def phase_plot(episodes, out_path):
    """Stacked bar per config: % of episodes that ended in each phase of the test."""
    percentages = {phase: [] for phase in PHASES}
    for config in CONFIGS:
        config_episodes = episodes[episodes["config"] == config]
        for phase in PHASES:
            percentages[phase].append(100 * np.mean(config_episodes["phase_of_failure"] == phase))

    plt.figure(figsize=(7, 4.5))
    bottom = np.zeros(len(CONFIGS))
    for phase in PHASES:
        plt.bar(CONFIGS, percentages[phase], bottom=bottom, label=PHASE_LABELS[phase], alpha=0.8)
        bottom += np.array(percentages[phase])
    plt.ylabel("% of episodes")
    plt.title("Phase in which episodes ended")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def theta_plot(per_seed, out_path):
    """Mean |theta| in the first 500 steps per config (mean +- std across seeds)."""
    means = [per_seed[per_seed["config"] == c]["mean_abs_theta"].mean() for c in CONFIGS]
    stds = [per_seed[per_seed["config"] == c]["mean_abs_theta"].std() for c in CONFIGS]

    plt.figure(figsize=(6, 4))
    plt.bar(CONFIGS, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel("Mean |theta| (rad), steps 1-500")
    plt.title("Pole angle during the default-force phase")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    """Make and save all figures."""
    parser = argparse.ArgumentParser(description="Plots of the evaluation results.")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    episodes = load_results(args.results_dir)
    for config in CONFIGS:
        if config in episodes["config"].values:
            bar_plot_config(episodes, config, os.path.join(args.results_dir, f"bar_{config}.png"))
    phase_plot(episodes, os.path.join(args.results_dir, "phase_of_failure.png"))
    theta_plot(per_seed_metrics(episodes), os.path.join(args.results_dir, "mean_abs_theta.png"))
    print(f"Figures saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
