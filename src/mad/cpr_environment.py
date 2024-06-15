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

    # Constants
    MIN_POOL_THRESHOLD: float = 5.0

    def __init__(self, conf_dict: Dict):
        """Initialize the CPR environment with configuration.
        
        Args:
            conf_dict: Dictionary containing environment configuration parameters.
        """
        self.spec = CprEnvironmentSpec(**conf_dict)
        self.pool = self.spec.max_resource_capacity

    def growth(self, resource_level: float) -> float:
        """Calculate resource growth using logistic growth model.
        
        Args:
            resource_level: Current resource level (N).
            
        Returns:
            Growth amount for the resource.
        """
        return self.spec.replenishment_rate * resource_level * (
            1 - resource_level / self.spec.max_resource_capacity
        )

    def harvest(self, total_effort: float, resource_level: float) -> float:
        """Calculate harvest level based on effort and resource level.
        
        Args:
            total_effort: Total effort from all agents (X).
            resource_level: Current resource level (N).
            
        Returns:
            Total harvest level.
        """
        return self.spec.beta * (total_effort ** self.spec.alpha) * (
            resource_level ** (1 - self.spec.alpha)
        )

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
        game_is_done = True if self.pool <= self.MIN_POOL_THRESHOLD else False

        # get feedback from env
        pis = [x / float(X) * harvest_level - self.spec.cost * x
               for x in xs]
        PI = sum(pis)

        # observations
        obs = [[x, pi, X, PI] for x, pi in zip(xs, pis)]

        return obs, self.reward(delta_n, pis), game_is_done

    def reset(self):
        """Reset the environment to initial state."""
        self.pool = self.spec.max_resource_capacity
