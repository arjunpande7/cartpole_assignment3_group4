"""
Reward functions used during training.

The reward is only used for LEARNING. Performance is always measured in the
number of steps survived, so all configs are compared on the same scale.
"""

THETA_MAX = 0.2095   # pole angle (rad) at which CartPole ends the episode (12 degrees)
X_MAX = 2.4          # cart position at which CartPole ends the episode


def compute_reward(next_state, config):
    """
    Return the reward for one step.

    Args:
        next_state: the observation after the step, [x, x_dot, theta, theta_dot].
        config: the config dict, with config["reward"]["type"] equal to
                "standard" or "stability" (and "lambda" for stability).

    standard:  1.0 for every step survived (normal CartPole reward).
    stability: 1 - lambda * 0.5 * (|theta| / THETA_MAX + |x| / X_MAX).
               Both terms are clipped to at most 1, so the reward stays in
               [1 - lambda, 1] = [0.5, 1] for lambda = 0.5. Surviving is
               therefore always better than dying.
    """
    reward_type = config["reward"]["type"]

    if reward_type == "standard":
        return 1.0

    if reward_type == "stability":
        lam = config["reward"]["lambda"]
        x = next_state[0]
        theta = next_state[2]
        angle_term = min(abs(theta) / THETA_MAX, 1.0)
        position_term = min(abs(x) / X_MAX, 1.0)
        return float(1.0 - lam * 0.5 * (angle_term + position_term))

    raise ValueError("Unknown reward type: " + str(reward_type))
