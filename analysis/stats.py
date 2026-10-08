"""
Statistical analysis of the evaluation results.

Reads every results/<config>_seed<N>_eval.csv made by evaluate.py.
The unit of analysis is ONE SEED (one trained network): per seed we compute one
number per metric, so each config has n = 5 and the design has n = 20.
Episodes are never treated as independent samples.

Tests (alpha = 0.05):
  - Two-way ANOVA (sampling x reward) per metric, with F, p and partial eta^2.
  - Assumption checks: Shapiro-Wilk on the residuals, Levene across the 4 groups.
    If a check fails, the ANOVA is repeated on log(metric + 1).
  - Supplementary: Mann-Whitney U, each strategy vs baseline, Holm-corrected.

Usage (from the repo root):  python -m analysis.stats [--results_dir results]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05
SHORT_POLE_MAX = 0.6
CONFIGS = ["baseline", "adaptive", "stability", "combined"]
# factor levels of each config in the 2x2 design
FACTORS = {
    "baseline":  {"sampling": "uniform",  "reward": "standard"},
    "adaptive":  {"sampling": "adaptive", "reward": "standard"},
    "stability": {"sampling": "uniform",  "reward": "stability"},
    "combined":  {"sampling": "adaptive", "reward": "stability"},
}
METRICS = ["overall_score", "short_pole_score", "pct_wind_fail", "mean_abs_theta"]
ANOVA_METRICS = ["overall_score", "short_pole_score", "pct_wind_fail"]


def load_results(results_dir):
    """Read all *_eval.csv files of the four configs into one DataFrame (one row per episode)."""
    frames = []
    for name in sorted(os.listdir(results_dir)):
        if name.endswith("_eval.csv"):
            frames.append(pd.read_csv(os.path.join(results_dir, name)))
    if len(frames) == 0:
        raise FileNotFoundError("No *_eval.csv files found in " + results_dir)
    episodes = pd.concat(frames, ignore_index=True)
    return episodes[episodes["config"].isin(CONFIGS)]


def mean_of_length_means(episodes):
    """Mean over pole lengths of the per-length mean steps (the official score)."""
    length_means = []
    for length in sorted(episodes["pole_length"].unique()):
        length_means.append(episodes[episodes["pole_length"] == length]["steps"].mean())
    return float(np.mean(length_means))


def per_seed_metrics(episodes):
    """Return one row per (config, seed) with every metric and the two factors."""
    rows = []
    for config in CONFIGS:
        config_episodes = episodes[episodes["config"] == config]
        for seed in sorted(config_episodes["seed"].unique()):
            run = config_episodes[config_episodes["seed"] == seed]
            short = run[run["pole_length"] <= SHORT_POLE_MAX + 1e-9]
            rows.append({
                "config": config,
                "seed": int(seed),
                "sampling": FACTORS[config]["sampling"],
                "reward": FACTORS[config]["reward"],
                "overall_score": mean_of_length_means(run),
                "short_pole_score": mean_of_length_means(short),
                "pct_wind_fail": 100 * float(np.mean(run["phase_of_failure"] == "wind")),
                "mean_abs_theta": float(run["mean_abs_theta"].mean()),
            })
    return pd.DataFrame(rows)


def two_way_anova(data, metric):
    """
    Two-way ANOVA: metric ~ sampling * reward.

    Returns:
        anova: table with F, p and partial eta^2 for sampling, reward and interaction.
        residuals: the model residuals (for the Shapiro-Wilk check).
    """
    model = smf.ols(f"{metric} ~ C(sampling) * C(reward)", data=data).fit()
    table = anova_lm(model, typ=2)
    ss_residual = table.loc["Residual", "sum_sq"]

    rows = []
    names = {"C(sampling)": "sampling", "C(reward)": "reward",
             "C(sampling):C(reward)": "interaction"}
    for term, name in names.items():
        ss = table.loc[term, "sum_sq"]
        rows.append({
            "metric": metric,
            "term": name,
            "F": table.loc[term, "F"],
            "p": table.loc[term, "PR(>F)"],
            "partial_eta_sq": ss / (ss + ss_residual),   # effect size
            "df": table.loc[term, "df"],
            "df_residual": table.loc["Residual", "df"],
        })
    return pd.DataFrame(rows), model.resid


def check_assumptions(data, metric, residuals):
    """Shapiro-Wilk (normal residuals) and Levene (equal variances across the 4 groups)."""
    shapiro_p = stats.shapiro(residuals).pvalue
    groups = [data[data["config"] == c][metric].values for c in CONFIGS]
    levene_p = stats.levene(*groups).pvalue
    return {"metric": metric, "shapiro_p": shapiro_p, "levene_p": levene_p,
            "passed": bool(shapiro_p >= ALPHA and levene_p >= ALPHA)}


def mann_whitney_vs_baseline(data, metric):
    """Mann-Whitney U of each strategy vs baseline (two-sided), Holm-corrected over the 3 tests."""
    baseline = data[data["config"] == "baseline"][metric].values
    rows = []
    for config in ["adaptive", "stability", "combined"]:
        strategy = data[data["config"] == config][metric].values
        result = stats.mannwhitneyu(strategy, baseline, alternative="two-sided")
        rows.append({"metric": metric, "comparison": config + " vs baseline",
                     "median_strategy": np.median(strategy), "median_baseline": np.median(baseline),
                     "U": result.statistic, "p": result.pvalue})
    table = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(table["p"], alpha=ALPHA, method="holm")
    table["p_holm"] = p_holm
    table["significant"] = reject
    return table


def describe(data):
    """Mean and std across seeds of every metric, per config."""
    rows = []
    for config in CONFIGS:
        config_data = data[data["config"] == config]
        row = {"config": config, "n_seeds": len(config_data)}
        for metric in METRICS:
            row[metric + "_mean"] = config_data[metric].mean()
            row[metric + "_std"] = config_data[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Run every test, save the tables to results/stats_*.csv and print a summary."""
    parser = argparse.ArgumentParser(description="Statistics on the evaluation results.")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    data = per_seed_metrics(load_results(args.results_dir))
    summary = describe(data)

    anova_tables, assumption_rows, mann_whitney_tables = [], [], []
    for metric in ANOVA_METRICS:
        anova, residuals = two_way_anova(data, metric)
        anova["scale"] = "raw"
        anova_tables.append(anova)
        check = check_assumptions(data, metric, residuals)
        check["scale"] = "raw"
        assumption_rows.append(check)

        if not check["passed"]:
            # log(metric + 1): the +1 keeps it defined when a metric is 0 (e.g. 0% wind failures)
            log_metric = "log_" + metric
            data[log_metric] = np.log(data[metric] + 1)
            log_anova, log_residuals = two_way_anova(data, log_metric)
            log_anova["scale"] = "log"
            anova_tables.append(log_anova)
            log_check = check_assumptions(data, log_metric, log_residuals)
            log_check["scale"] = "log"
            assumption_rows.append(log_check)

        mann_whitney_tables.append(mann_whitney_vs_baseline(data, metric))

    anova_all = pd.concat(anova_tables, ignore_index=True)
    assumptions = pd.DataFrame(assumption_rows)
    mann_whitney_all = pd.concat(mann_whitney_tables, ignore_index=True)

    data.to_csv(os.path.join(args.results_dir, "stats_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(args.results_dir, "stats_summary.csv"), index=False)
    anova_all.to_csv(os.path.join(args.results_dir, "stats_anova.csv"), index=False)
    assumptions.to_csv(os.path.join(args.results_dir, "stats_assumptions.csv"), index=False)
    mann_whitney_all.to_csv(os.path.join(args.results_dir, "stats_mannwhitney.csv"), index=False)

    pd.set_option("display.width", 160)
    print("=== Mean (std) across seeds per config ===")
    for _, row in summary.iterrows():
        parts = [f"{m} {row[m + '_mean']:.1f} ({row[m + '_std']:.1f})" if m != "mean_abs_theta"
                 else f"{m} {row[m + '_mean']:.4f} ({row[m + '_std']:.4f})" for m in METRICS]
        print(f"{row['config']:10s} n={row['n_seeds']}  " + " | ".join(parts))

    print("\n=== Two-way ANOVA (sampling x reward) ===")
    for _, row in anova_all.iterrows():
        mark = "*" if row["p"] < ALPHA else " "
        print(f"{row['metric']:22s} [{row['scale']}] {row['term']:12s} "
              f"F({row['df']:.0f},{row['df_residual']:.0f}) = {row['F']:7.2f}  "
              f"p = {row['p']:.4f}{mark}  partial eta^2 = {row['partial_eta_sq']:.3f}")

    print("\n=== Assumption checks (p < 0.05 = assumption violated) ===")
    for _, row in assumptions.iterrows():
        print(f"{row['metric']:22s} [{row['scale']}] Shapiro p = {row['shapiro_p']:.3f}  "
              f"Levene p = {row['levene_p']:.3f}  {'OK' if row['passed'] else 'VIOLATED'}")

    print("\n=== Mann-Whitney U vs baseline (Holm-corrected) ===")
    for _, row in mann_whitney_all.iterrows():
        print(f"{row['metric']:18s} {row['comparison']:22s} U = {row['U']:5.1f}  p = {row['p']:.4f}  "
              f"p_holm = {row['p_holm']:.4f}  {'significant' if row['significant'] else 'not significant'}")

    print(f"\nTables saved to {args.results_dir}/stats_*.csv")
    print("Note: with 5 seeds per config the power is limited; a non-significant result "
          "means 'no clear evidence', not 'no effect'.")


if __name__ == "__main__":
    main()
