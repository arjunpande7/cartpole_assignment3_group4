"""
Quick checks for src/rewards.py.

Run from the repo root:  python -m tests.test_rewards
"""
from src.rewards import compute_reward

STANDARD = {"reward": {"type": "standard"}}
STABILITY = {"reward": {"type": "stability", "lambda": 0.5}}


def test_standard():
    """The standard reward is always 1."""
    assert compute_reward([0.0, 0.0, 0.0, 0.0], STANDARD) == 1.0
    assert compute_reward([2.0, 1.0, 0.2, 1.0], STANDARD) == 1.0
    print("standard reward: OK")


def test_stability():
    """The stability reward is 1 when centred and upright and drops to 0.5 at the limits."""
    assert compute_reward([0.0, 0.0, 0.0, 0.0], STABILITY) == 1.0
    # at both limits: 1 - 0.5 * 0.5 * (1 + 1) = 0.5
    assert abs(compute_reward([2.4, 0.0, 0.2095, 0.0], STABILITY) - 0.5) < 1e-9
    # the sign does not matter
    assert compute_reward([-1.2, 0.0, -0.1, 0.0], STABILITY) == compute_reward([1.2, 0.0, 0.1, 0.0], STABILITY)
    # halfway on both: 1 - 0.25 * (0.5 + 0.5) = 0.75
    assert abs(compute_reward([1.2, 0.0, 0.10475, 0.0], STABILITY) - 0.75) < 1e-9
    # beyond the limits the terms are clipped, so the reward never goes below 0.5
    assert compute_reward([5.0, 0.0, 1.0, 0.0], STABILITY) == 0.5
    print("stability reward: OK")


if __name__ == "__main__":
    test_standard()
    test_stability()
    print("\nAll reward tests passed.")
