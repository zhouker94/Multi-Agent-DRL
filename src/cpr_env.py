import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


@dataclass
class CPRConfig:
    max_steps: int
    replenishment_rate: float
    max_resource_capacity: float  # maximum resource capacity
    alpha: float  # parameter alpha
    beta: float  # parameter beta
    cost: float  # cost


class CPREnvironment(py_environment.PyEnvironment):

    def set_state(self, state: Any) -> None:
        pass

    def get_state(self) -> None:
        pass

    def get_info(self) -> None:
        pass

    def __init__(self, num_player: int, s_weight: float):
        super().__init__()

        self._num_player = num_player
        self._s_weight = s_weight
        self._observation_size = num_player + 1

        # Game Config
        with open('config/environment.json', 'r') as f:
            conf = json.load(f)
            self._conf = CPRConfig(**conf)

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._observation_size,),
            dtype=np.float,
            name='observation'
        )

        self._reward_spec = array_spec.ArraySpec(
            shape=(self._num_player,),
            dtype=np.float,
            name='reward'
        )

        self._step_count = 0
        self._state = self._conf.max_resource_capacity
        self._episode_ended = False

    def action_spec(self):
        return array_spec.BoundedArraySpec(
            shape=(self._num_player,),
            dtype=np.float,
            minimum=0,
            maximum=1,
            name='action'
        )

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self._reward_spec

    def _reset(self):
        self._step_count = 0
        self._state = self._conf.max_resource_capacity
        self._episode_ended = False
        return ts.restart(np.zeros(self._observation_size),
                          reward_spec=np.zeros(self._num_player))

    def _step(self, action: np.ndarray):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        total_invest = np.sum(action)
        harvest_level = self.harvest(total_invest)
        delta_n = self.growth - harvest_level
        self._state += delta_n

        obs = np.zeros(self._observation_size)

        # Make sure episodes don't go on forever.
        if self._state <= 0 or self._step_count >= self._conf.max_steps:
            self._episode_ended = True
            return ts.termination(
                observation=obs,
                reward=np.zeros(self._num_player),
                outer_dims=()
            )

        pis = action / total_invest * harvest_level - self._conf.cost * action

        # lambda_ -> sustainability_goal
        lambda_ = 1 if delta_n >= 0 else -1
        # Pi's -> wealth_goal
        reward = self._s_weight * lambda_ + (1 - self._s_weight) * pis

        obs[:-1] = pis
        obs[-1] = self._state

        self._step_count += 1

        return ts.transition(obs, reward=reward, outer_dims=())

    @property
    def growth(self):
        return self._conf.replenishment_rate * self._state * (1 - self._state / self._conf.max_resource_capacity)

    def harvest(self, x):
        return self._conf.beta * (x ** self._conf.alpha) * (self._state ** (1 - self._conf.alpha))


if __name__ == "__main__":
    environment = CPREnvironment(num_player=5, s_weight=0.5)
    utils.validate_py_environment(environment, episodes=5)
