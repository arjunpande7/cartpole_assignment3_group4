"""
Pole-length samplers used during training.

Every training episode the training loop asks a sampler for a pole length
(.sample()) and, after the episode, tells the sampler how long the agent
survived (.update()). Two samplers exist:

- UniformSampler:  every length in [low, high] is equally likely (baseline).
- AdaptiveSampler: lengths the agent currently struggles with are picked more
                   often, inspired by prioritized level replay (Jiang et al., 2021).

Use make_sampler(config, rng) to build the sampler named in the config.
"""
import numpy as np


class UniformSampler:
    """Samples every pole length in [low, high] with equal probability."""

    def __init__(self, low, high, rng, n_bins=15):
        """
        Args:
            low, high: the range of pole lengths.
            rng: a np.random.Generator, so results can be reproduced with a seed.
            n_bins: only used by get_distribution(), so the log has the same
                    shape as the adaptive sampler's log.
        """
        self.low = low
        self.high = high
        self.rng = rng
        self.n_bins = n_bins

    def sample(self):
        """Return the pole length for the next episode."""
        return float(self.rng.uniform(self.low, self.high))

    def update(self, length, episode_steps):
        """Uniform sampling does not learn from results, so nothing happens here."""
        pass

    def get_distribution(self):
        """Return the probability of each bin: all bins are equally likely."""
        return np.ones(self.n_bins) / self.n_bins


class AdaptiveSampler:
    """
    Picks pole lengths where the agent currently fails more often.

    The range [low, high] is split into n_bins equal-width bins. For every bin we
    keep an exponential moving average (EMA) of the episode length. A bin where
    the agent dies quickly is "difficult" and gets a higher sampling probability:

        difficulty d_b = 1 - min(EMA_b / cap, 1)
        P(b) = mix * (1 / n_bins) + (1 - mix) * softmax(d_b / temperature)

    The 'mix' part keeps some uniform sampling so easy lengths are not forgotten.
    """

    def __init__(self, low, high, n_bins, mix, temperature, alpha, cap, rng):
        """
        Args:
            low, high: the range of pole lengths.
            n_bins: number of equal-width bins.
            mix: share of uniform sampling (0.2 = 20%).
            temperature: softmax temperature; lower = focus more on hard bins.
            alpha: EMA smoothing; how strongly a new episode changes the estimate.
            cap: the training episode cap; surviving this long means difficulty 0.
            rng: a np.random.Generator, so results can be reproduced with a seed.
        """
        self.low = low
        self.high = high
        self.n_bins = n_bins
        self.mix = mix
        self.temperature = temperature
        self.alpha = alpha
        self.cap = cap
        self.rng = rng

        self.edges = np.linspace(low, high, n_bins + 1)   # bin borders
        self.ema = np.full(n_bins, np.nan)                 # nan = no estimate yet
        # Warm-up: the first n_bins samples visit every bin once, in random order
        self.warmup_order = list(rng.permutation(n_bins))

    def bin_of(self, length):
        """Return the index of the bin that a pole length falls in."""
        width = (self.high - self.low) / self.n_bins
        index = int((length - self.low) / width)
        # the very top value (length == high) belongs to the last bin
        return min(max(index, 0), self.n_bins - 1)

    def get_distribution(self):
        """Return the current sampling probability of every bin."""
        difficulty = np.ones(self.n_bins)          # bins without estimate count as hard
        for b in range(self.n_bins):
            if not np.isnan(self.ema[b]):
                difficulty[b] = 1 - min(self.ema[b] / self.cap, 1)

        weights = np.exp(difficulty / self.temperature)
        softmax = weights / np.sum(weights)
        return self.mix * (1 / self.n_bins) + (1 - self.mix) * softmax

    def sample(self):
        """Return the pole length for the next episode."""
        if len(self.warmup_order) > 0:
            b = self.warmup_order.pop(0)                         # warm-up phase
        else:
            b = self.rng.choice(self.n_bins, p=self.get_distribution())
        # a uniformly random length inside the chosen bin
        return float(self.rng.uniform(self.edges[b], self.edges[b + 1]))

    def update(self, length, episode_steps):
        """Update the EMA of the bin that 'length' belongs to with the episode length."""
        b = self.bin_of(length)
        if np.isnan(self.ema[b]):
            self.ema[b] = episode_steps                          # first result for this bin
        else:
            self.ema[b] = (1 - self.alpha) * self.ema[b] + self.alpha * episode_steps


def make_sampler(config, rng):
    """
    Build the sampler described in config["sampler"].

    Args:
        config: the config dict loaded from YAML, with a "sampler" section.
        rng: a np.random.Generator.
    Returns:
        A UniformSampler or AdaptiveSampler.
    """
    s = config["sampler"]
    if s["type"] == "uniform":
        return UniformSampler(s["low"], s["high"], rng, s.get("n_bins", 15))
    if s["type"] == "adaptive":
        return AdaptiveSampler(s["low"], s["high"], s["n_bins"], s["mix"],
                               s["temperature"], s["alpha"], s["cap"], rng)
    raise ValueError("Unknown sampler type: " + str(s["type"]))
