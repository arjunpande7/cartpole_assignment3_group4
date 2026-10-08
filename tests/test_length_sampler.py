"""
Quick checks for src/length_sampler.py.

Run from the repo root:  python -m tests.test_length_sampler
"""
import numpy as np

from src.length_sampler import UniformSampler, AdaptiveSampler, make_sampler

CONFIG_ADAPTIVE = {"sampler": {"type": "adaptive", "low": 0.4, "high": 1.8, "n_bins": 15,
                               "mix": 0.2, "temperature": 0.2, "alpha": 0.1, "cap": 2000}}
CONFIG_UNIFORM = {"sampler": {"type": "uniform", "low": 0.4, "high": 1.8}}


def test_uniform():
    """Uniform lengths stay inside [low, high] and the distribution is flat."""
    sampler = make_sampler(CONFIG_UNIFORM, np.random.default_rng(0))
    assert isinstance(sampler, UniformSampler)
    lengths = [sampler.sample() for _ in range(1000)]
    assert min(lengths) >= 0.4 and max(lengths) <= 1.8
    distribution = sampler.get_distribution()
    assert np.allclose(distribution, 1 / 15) and np.isclose(np.sum(distribution), 1)
    print("uniform sampler: OK")


def test_warmup():
    """The first 15 samples visit every bin exactly once."""
    sampler = make_sampler(CONFIG_ADAPTIVE, np.random.default_rng(0))
    assert isinstance(sampler, AdaptiveSampler)
    bins = []
    for _ in range(15):
        length = sampler.sample()
        bins.append(sampler.bin_of(length))
        sampler.update(length, 100)
    assert sorted(bins) == list(range(15)), bins
    print("adaptive warm-up visits every bin once: OK")


def test_focus_on_hard_lengths():
    """
    Short poles (< 1.1) always fail after 20 steps, long poles survive 2000.
    The sampler should then pick short-pole bins much more often.
    """
    sampler = make_sampler(CONFIG_ADAPTIVE, np.random.default_rng(1))
    short_count = 0
    n_episodes = 3000
    for episode in range(n_episodes):
        length = sampler.sample()
        if length < 1.1:
            steps = 20
            short_count += 1
        else:
            steps = 2000
        sampler.update(length, steps)

    distribution = sampler.get_distribution()
    short_share = short_count / n_episodes
    print(f"share of episodes with a short pole: {short_share:.2f} (uniform would be 0.50)")
    print("final bin probabilities:", np.round(distribution, 3))
    assert np.isclose(np.sum(distribution), 1)
    # expected about 0.2 * 0.5 (uniform part) + 0.8 * ~1 (softmax part) = ~0.9
    assert short_share > 0.8
    # mastered bins only get the uniform share: 0.2 / 15 plus a tiny softmax share
    assert distribution[-1] < 0.02
    print("adaptive sampler focuses on hard lengths: OK")


def test_bin_of_edges():
    """The lowest and highest lengths fall in the first and last bin."""
    sampler = make_sampler(CONFIG_ADAPTIVE, np.random.default_rng(0))
    assert sampler.bin_of(0.4) == 0
    assert sampler.bin_of(1.8) == 14
    print("bin edges: OK")


if __name__ == "__main__":
    test_uniform()
    test_warmup()
    test_focus_on_hard_lengths()
    test_bin_of_edges()
    print("\nAll length sampler tests passed.")
