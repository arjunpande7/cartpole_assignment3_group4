import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

configs = ["baseline", "adaptive", "stability", "combined"]
seeds = [0, 1, 2, 3, 4]


def get_factors(configuration):
    if configuration == "baseline":
        return "uniform", "standard"
    if configuration == "adaptive":
        return "adaptive", "standard"
    if configuration == "stability":
        return "uniform", "stability"
    return "adaptive", "stability"


def smooth(values, amount):
    return pd.Series(values).rolling(amount, min_periods=1).mean()


def plot_training_curves(configs, seeds):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(len(configs)):
        config = configs[i]
        ax = axes[i]

        for seed in seeds:
            log = pd.read_csv(f"results/train_logs/{config}_seed{seed}.csv")
            ax.plot(log["total_steps"], smooth(log["episode_length"], 100), linewidth=1, label=f"seed {seed}")

        ax.set_title(config)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode length (mean of 100)")
        ax.grid(True)

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    print("Saved results/training_curves.png")


def collect_metrics(configs, seeds):
    rows = []

    for config in configs:
        sampling, reward = get_factors(config)

        for seed in seeds:
            log = pd.read_csv(f"results/train_logs/{config}_seed{seed}.csv")
            last = log[log["total_steps"] > 900000]
            last100k = last["episode_length"].mean()

            evaluation = pd.read_csv(f"results/{config}_seed{seed}_eval.csv")
            past_1000 = (evaluation["steps"] > 1000).mean() * 100

            rows.append({
                "config": config,
                "seed": seed,
                "sampling": sampling,
                "reward": reward,
                "last100k": last100k,
                "log_last100k": np.log(last100k),
                "pct_past_1000": past_1000,
            })

    return pd.DataFrame(rows)


def print_summary(df, configs):
    print("\n=== Mean across seeds per config")
    for config in configs:
        sub = df[df["config"] == config]
        print(f"{config:10s} last100k = {sub['last100k'].mean():7.1f} ({sub['last100k'].std():6.1f}) | "
              f"past 1000 = {sub['pct_past_1000'].mean():5.1f}% ({sub['pct_past_1000'].std():4.1f})")


def two_way_anova(df, metric):
    model = ols(f"{metric} ~ C(sampling) * C(reward)", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    residual = table.loc["Residual", "sum_sq"]

    print(f"\n=== Two-way ANOVA on {metric} ===")
    for term in ["C(sampling)", "C(reward)", "C(sampling):C(reward)"]:
        F = table.loc[term, "F"]
        p = table.loc[term, "PR(>F)"]
        eta = table.loc[term, "sum_sq"] / (table.loc[term, "sum_sq"] + residual)
        if p < 0.05:
            star = "*"
        else:
            star = ""
        print(f"{term:25s} F = {F:6.2f}  p = {p:.4f}{star}  partial eta^2 = {eta:.3f}")


plot_training_curves(configs, seeds)

df = collect_metrics(configs, seeds)
df.to_csv("results/stats_training_metrics.csv", index=False)

print_summary(df, configs)
two_way_anova(df, "log_last100k")
two_way_anova(df, "pct_past_1000")