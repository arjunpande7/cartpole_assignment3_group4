# Training One DQN Agent for Many Pole Lengths

**Introduction to Reinforcement Learning, Assignment 3, Team 4 (Vrije Universiteit Amsterdam)**
Arjun Pande · Arian Bahreininejad · Saransh Khatri

This repository contains the code, trained networks and results for our paper. One Deep Q-Network (DQN) agent is trained on `CartPole-v1` (OpenAI Gym) to balance the pole for every pole length between 0.4 and 1.8. During training only the pole length is varied. The test script also raises the force on the cart sharply after 500 steps and keeps increasing it after 1000 steps. The agent never experiences these forces during training.

## Research question

> How do adaptive pole-length sampling and a reward that encourages a stability margin affect a DQN agent's ability to generalise across pole lengths and to survive sudden increases in force it never encountered during training, and are the two effects complementary?

**Hypotheses**

- **H1:** adaptive sampling mainly helps on short poles.
- **H2:** the stability reward mainly improves survival of the force increase, because the agent keeps a margin.
- **H3:** the two effects are complementary.

## Experimental design

Crossing two changes to the training procedure gives a 2×2 design. Each configuration is trained with 5 seeds (0–4), for 20 runs in total.

| Config | Pole-length sampling | Reward |
|---|---|---|
| `baseline` | Uniform | Standard (+1 per step) |
| `adaptive` | Adaptive | Standard |
| `stability` | Uniform | Stability |
| `combined` | Adaptive | Stability |

**Uniform sampling.** At the start of every episode the pole length is drawn uniformly from [0.4, 1.8].

**Adaptive sampling** (`src/length_sampler.py`), inspired by Prioritized Level Replay (Jiang et al., 2021):

- The range [0.4, 1.8] is split into B = 15 equal-width bins.
- Each bin keeps an exponential moving average L̄_b of the episode lengths reached in that bin (α = 0.1).
- The difficulty of a bin is d_b = 1 − min(L̄_b / C, 1), with C = 2000 (the training episode cap).
- A bin is chosen with P(b) = μ · 1/B + (1 − μ) · softmax(d_b / τ), with μ = 0.2 and τ = 0.2. The pole length is then drawn uniformly within that bin.
- During the first 15 episodes each bin is visited once, in random order.

**Stability reward** (`src/rewards.py`): r = 1 − λ · ½ · (min(|θ|/θ_max, 1) + min(|x|/x_max, 1)), with θ_max = 0.2095 rad (12°), x_max = 2.4 and λ = 0.5.

- The reward is 1 when the pole is upright and the cart is centred, and 0.5 at the limits, so surviving is always better than failing.
- Unlike potential-based shaping (Ng et al., 1999), it deliberately changes the optimal policy towards one that keeps a margin.
- It is used for learning only. All configurations are evaluated on steps survived.

### DQN settings

The network is the fixed `QNetwork` from `test_script.py` (4 → 128 → 128 → 2, ReLU). All four configurations use the same settings, stored in `config/*.yaml`:

| Setting | Value |
|---|---|
| Optimiser, learning rate | Adam, 5·10⁻⁴ |
| Discount factor γ | 0.99 |
| Loss, gradient clipping | Huber (Smooth L1), norm 10 |
| Replay buffer, batch size | 100,000 transitions, 64 |
| Learning starts after | 5,000 transitions |
| Updates | 1 gradient step per environment step |
| Target network update | Hard copy every 1,000 steps |
| Exploration | ε-greedy, 1.0 → 0.05, linear over the first 100,000 steps |
| Training budget | 1,000,000 environment steps |
| Episode cap (training) | 2,000 steps; reaching the cap is not treated as terminal |

A pilot run with the baseline configuration (seed 0) determined the 1M-step budget. After 300,000 steps it averaged 260 test steps, and 99% of test episodes ended before step 500. After 1M steps it averaged 538, and 96% of test episodes reached the force increase. The **final weights** of every run are used, with no checkpoint selection.

### Evaluation

`evaluate.py` follows the procedure of the official `test_script.py`:

- 30 pole lengths (`np.linspace(0.4, 1.8, 30)`), 10 episodes per length, greedy actions.
- An episode ends when the pole passes 12° or the cart leaves the track.
- The force schedule is the same. Steps 1–500 use the default force of 10, the force becomes 75 at step 500, and after step 1000 it is 25 + 0.01·t.

There is one change: episode *i* at every pole length starts from reset seed 1000 + *i*, so all networks face the same initial states. The mean |θ| over the first 500 steps is also recorded. The best network is additionally run through the **unmodified** official test script.

### Metrics and statistics

The unit of analysis is a trained network (n = 5 per configuration), not an episode. Per network we compute:

- **Score:** the mean of the 30 per-length average episode lengths (the official metric).
- **Short:** the same average, restricted to pole lengths ≤ 0.6.
- **Past 1000:** the % of test episodes that survive past step 1000, i.e. through the high-force phase.
- **|θ|:** the mean pole angle during steps 1–500.
- **Last 100k:** the mean training episode length over the final 100,000 training steps.

We had planned to measure the % of episodes that *ended* in the high-force phase. We replaced it with "Past 1000" because the original is confounded: an agent that fails at step 13 scores 0% on it, the same as one that survives the phase.

For each metric we fit a two-way ANOVA (sampling × reward) and report F, p and partial η²:
- **Assumption checks:** Shapiro–Wilk tests the normality of the residuals and Levene's test the equality of variances. If either fails, the ANOVA is repeated on log(metric + 1).
- **Comparisons with the baseline:** each strategy is compared with a two-sided Mann–Whitney U test, Holm-corrected over the three comparisons.
- **Significance level:** α = 0.05.

## Results

Mean (standard deviation) over the five seeds:

| Config | Score | Median score | Best seed | Short | Past 1000 (%) | \|θ\| (rad) | Last 100k |
|---|---|---|---|---|---|---|---|
| Baseline | 518 (659) | 183 | 1,652 | 298 (282) | 4.7 (9.9) | 0.100 (0.027) | 228 (94) |
| Adaptive | 1,132 (2,340) | 96 | 5,316 | 300 (550) | 14.6 (32.6) | 0.084 (0.043) | 113 (72) |
| Stability | 6,508 (9,215) | 624 | 21,032 | 5,543 (7,157) | 40.1 (51.1) | 0.040 (0.036) | 1,070 (556) |
| Combined | 6,488 (5,901) | 8,697 | 12,622 | 2,503 (2,323) | 54.7 (49.9) | 0.024 (0.027) | 1,012 (501) |

- **Overall score:** the reward has a significant effect (log scale: F(1,16) = 5.85, p = 0.028, partial η² = 0.27). Sampling (p = 0.89) and the interaction (p = 0.56) do not.
- **Short poles:** the reward effect is larger (F(1,16) = 11.05, p = 0.004, partial η² = 0.41). Adaptive sampling has no effect (300 vs. 298, p = 0.43).
- **Surviving the force increase:** the reward effect is just significant (F(1,16) = 4.55, p = 0.049, partial η² = 0.22).
- **Training:** the reward effect on the last-100k episode length is large (F(1,16) = 27.2, p < 0.001, partial η² = 0.63). The residuals are not normal even on the log scale, so this result should be read with caution.
- **Mann–Whitney tests:** none of the comparisons with the baseline is significant after Holm correction.
- **Best network:** `stability_seed2` scores 21,032 in our evaluation and 20,791 under the official test script, which does not fix the initial states.

**Conclusion.** The stability reward is the only factor with a clear effect, which supports H2. H1 and H3 are not supported. Results differ strongly between seeds, partly because the final weights are used and several runs collapse during training. See the paper for the full discussion.

## Repository structure

```
├── test_script.py            official test script (test logic unchanged; trained_nn set to the best network)
├── run_all.py                trains all four configs for the given seeds, several runs in parallel
├── evaluate.py               test procedure with fixed reset seeds + extra metrics -> results/*_eval.csv
├── requirements.txt
├── config/                   baseline.yaml, adaptive.yaml, stability.yaml, combined.yaml
├── src/
│   ├── network.py            QNetwork (copied from test_script.py)
│   ├── replay_buffer.py      experience replay buffer
│   ├── agent.py              DQN agent: epsilon-greedy actions, Huber loss, target network
│   ├── train.py              training loop for one config and one seed
│   ├── length_sampler.py     UniformSampler and AdaptiveSampler
│   └── rewards.py            standard and stability reward
├── analysis/
│   ├── stats.py              per-seed metrics, two-way ANOVA, assumption checks, Mann–Whitney + Holm
│   ├── plots.py              per-config bar plots, phase-of-failure plot, |theta| plot
│   ├── training_curves.py    training curves, "Past 1000" and "Last 100k" metrics with their ANOVAs
│   └── make_fake_results.py  fake evaluation files for testing the analysis code
├── tests/                    checks for the samplers and the rewards
├── weights/                  final weights of all 20 runs: <config>_seed<N>.pth
└── results/
    ├── <config>_seed<N>_eval.csv   one row per test episode (20 files)
    ├── train_logs/                 one training log per run (20 files)
    ├── stats_*.csv                 statistics tables
    ├── training_curves.png         training curves of all 20 runs (Figure 1)
    ├── bar_<config>.png            per-length test scores per config (Figure 2)
    └── official/bar_plot.png       best network under the official test script (Figure 2)
```

## Setup

The code needs Python 3.10, `gym==0.26.2` (not gymnasium, because the test script uses `import gym`) and `numpy<2`, because gym 0.26 does not work with NumPy 2.

```bash
git clone https://github.com/arjunpande7/cartpole_assignment3_group4
cd cartpole_assignment3_group4
conda create -n cartpole python=3.10 -y
conda activate cartpole
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Reproducing the results

**1. Train** all 20 runs. Each run takes about 14 minutes on a laptop CPU with 6 runs in parallel.

```bash
mkdir -p runs                                    # run_all.py writes its log files here
python run_all.py --seeds 0 1 2 3 4 --parallel 6
```

To train one configuration and one seed:

```bash
python -m src.train --config config/stability.yaml --seed 2
```

Weights are saved to `weights/<config>_seed<N>.pth` and training logs to `runs/<config>_seed<N>/train_log.csv`. The logs used in the paper are copied to `results/train_logs/`.

**2. Evaluate** every network. Good agents survive tens of thousands of steps, so this can take a while.

```bash
python evaluate.py --all
# or a single network:
python evaluate.py --weights weights/stability_seed2.pth --config stability --seed 2 --out results/stability_seed2_eval.csv
```

**3. Analyse.**

```bash
python -m analysis.stats               # Score, Short, |theta| per seed + ANOVAs -> results/stats_*.csv
python -m analysis.plots               # bar plots -> results/bar_*.png
python analysis/training_curves.py     # training curves, Past 1000 and Last 100k + their ANOVAs
```

**4. Official test script.** `trained_nn` in `test_script.py` is set to the best network (`stability_seed2.pth`). Run:

```bash
python test_script.py
```

It shows and saves `bar_plot.png` and writes `experiment_results.xlsx`.

**Checks** (optional):

```bash
python -m tests.test_length_sampler
python -m tests.test_rewards
```

