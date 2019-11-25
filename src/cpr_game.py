#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/22 10:16
# @Author  : Hanwei Zhu
# @File    : environment.py


def simple_step_function(X):
    if X > 0:
        return 1
    elif X == 0:
        return 0
    else:
        return -1


class CPRGame:
    """CPR game
    """

    def __init__(self, conf):
        self.spec = {
            "W": conf["sustainable_weight"],  # weight of sustainability
            "RG": conf["replenishment_rate"],  # replenishment rate
            "MRC": conf["max_resource_capacity"],  # maximum resource capacity
            "ALPHA": conf["alpha"],  # parameter alpha
            "BETA": conf["beta"],  # parameter beta
            "COST": conf["cost"],  # cost
        }
        self.pool = conf["max_resource_capacity"]

    def growth(self, N):
        return self.spec["RG"] * N * (1 - N / self.spec["MRC"])

    def harvest(self, x, N):
        return self.spec["BETA"] * (x ** self.spec["ALPHA"]) \
            * (N ** (1 - self.spec["ALPHA"]))

    def reward(self, delta_n, pis, step_func=simple_step_function):
        rewards = []
        # lambda_ -> sustainability_goal
        lambda_ = step_func(delta_n)
        # xi -> wealth_goal
        for pi in pis:
            xi = step_func(pi)
            rewards.append(self.spec["W"] * lambda_ +
                           (1 - self.spec["W"]) * xi)

        return rewards

    def step(self, xs):
        # Change game status
        X = sum(xs)
        harvest_level = self.harvest(X, self.pool)
        delta_n = self.growth(self.pool) - harvest_level
        self.pool += delta_n
        game_is_done = True if self.pool <= 5.0 else False

        # get feedback from env
        pis = [x / float(X) * harvest_level - self.spec["COST"] * x
               for x in xs]
        PI = sum(pis)

        # observations
        obs = [[x, pi, X, PI] for x, pi in zip(xs, pis)]

        return obs, self.reward(delta_n, pis), game_is_done

    def reset(self):
        self.pool = self.spec["MRC"]
