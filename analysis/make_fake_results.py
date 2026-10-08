"""
Generate FAKE evaluation CSVs to test stats.py and plots.py before real results exist.

The files have exactly the format of evaluate.py (4 configs x 5 seeds, 30 pole
lengths x 10 episodes). The fake configs differ on purpose so the effects show:
  - adaptive sampling: short poles fail less often in the normal phase
  - stability reward:  fewer failures during the wind phase, smaller |theta|
  - combined:          both

They are written to results/fake/ by default, so they never mix with real results.

Usage (from the repo root):
  python -m analysis.make_fake_results
  python -m analysis.stats --results_dir results/fake
  python -m analysis.plots --results_dir results/fake
"""
import argparse
import os

import numpy as np
import pandas as pd

from analysis.stats import FACTORS

POLE_LENGTHS = np.linspace(0.4, 1.8, 30)


def fake_episode(rng, length, sampling, reward, seed_skill):
    """Return (steps, mean_abs_theta, mean_abs_x) of one made-up episode."""
    # chance of failing in the normal phase: high for short poles
    p_normal = max(0.05, 0.9 - 0.8 * (length - 0.4) / 0.6) if length < 1.0 else 0.05
    if sampling == "adaptive":
        p_normal *= 0.5
    # chance of failing in the wind phase (given it survived the normal phase)
    p_wind = 0.6 if reward == "standard" else 0.25
    p_normal = min(1, max(0, p_normal + seed_skill))
    p_wind = min(1, max(0, p_wind + seed_skill))

    if rng.random() < p_normal:
        steps = int(rng.integers(20, 501))
    elif rng.random() < p_wind:
        steps = int(rng.integers(501, 1001))
    else:
        steps = int(rng.integers(1001, 20000))

    theta_level = 0.010 if reward == "standard" else 0.005
    mean_abs_theta = abs(rng.normal(theta_level, 0.002))
    mean_abs_x = abs(rng.normal(0.3 if reward == "standard" else 0.15, 0.05))
    return steps, mean_abs_theta, mean_abs_x


def main():
    """Write one fake *_eval.csv per config and seed."""
    parser = argparse.ArgumentParser(description="Generate fake evaluation results for testing.")
    parser.add_argument("--out_dir", default=os.path.join("results", "fake"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    for config, factors in FACTORS.items():
        for seed in range(5):
            seed_skill = rng.normal(0, 0.05)       # some networks are a bit better than others
            rows = []
            for length in POLE_LENGTHS:
                for episode in range(10):
                    steps, theta, x = fake_episode(rng, length, factors["sampling"],
                                                   factors["reward"], seed_skill)
                    if steps <= 500:
                        phase = "normal"
                    elif steps <= 1000:
                        phase = "wind"
                    else:
                        phase = "increasing"
                    rows.append({"config": config, "seed": seed, "pole_length": round(float(length), 4),
                                 "episode": episode, "steps": steps, "phase_of_failure": phase,
                                 "mean_abs_theta": theta, "mean_abs_x": x})
            path = os.path.join(args.out_dir, f"{config}_seed{seed}_eval.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Fake results written to {args.out_dir}/")


if __name__ == "__main__":
    main()
