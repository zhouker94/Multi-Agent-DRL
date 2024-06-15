from typing import Dict

from pydantic import BaseModel


class CprEnvironmentSpec(BaseModel):
    """Class for CPR environment Specification"""
    sustainable_weight: float
    replenishment_rate: float
    max_resource_capacity: float
    alpha: float
    beta: float
    cost: float


def step_func(x: float):
    """A step function returns 1 if the input x is greater than 0, otherwise returns -1.

    Parameters:
    x (float): The input value to evaluate.

    Returns:
    int: 1 if x > 0, -1 if x <= 0.
    """
    return 1 if x > 0 else -1


class CprEnvironment:
    """This class represents a Common Pool Resource Environment"""

    def __init__(self, conf_dict: Dict):
        self.spec = CprEnvironmentSpec(**conf_dict)
        self.pool = self.spec.max_resource_capacity

    def growth(self, N):
        return self.spec.replenishment_rate * N * (1 - N / self.spec.max_resource_capacity)

    def harvest(self, x, N):
        return self.spec.beta * (x ** self.spec.alpha) \
            * (N ** (1 - self.spec.alpha))

    def reward(self, delta_n, pis):
        # lambda_ -> sustainability_goal
        lambda_ = step_func(delta_n)
        # xi -> wealth_goal
        xis = map(step_func, pis)
        return [self.spec.sustainable_weight * lambda_ + (1 - self.spec.sustainable_weight) * xi for xi in xis]

    def step(self, xs):
        # Change game status
        X = sum(xs)
        harvest_level = self.harvest(X, self.pool)
        delta_n = self.growth(self.pool) - harvest_level
        self.pool += delta_n
        game_is_done = True if self.pool <= 5.0 else False

        # get feedback from env
        pis = [x / float(X) * harvest_level - self.spec.cost * x
               for x in xs]
        PI = sum(pis)

        # observations
        obs = [[x, pi, X, PI] for x, pi in zip(xs, pis)]

        return obs, self.reward(delta_n, pis), game_is_done

    def reset(self):
        self.pool = self.spec.max_resource_capacity
