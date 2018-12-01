#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/22 10:16
# @Author  : Hanwei Zhu
# @File    : environment.py


class CPRGame(object):
    def __init__(self, conf):
        self.W = conf["sustainable_weight"]
        
        self.RG = conf["replenishment_rate"]
        self.MRC = conf["max_resource_capacity"]
        self.ALPHA = conf["alpha"]
        self.BETA = conf["beta"]
        self.COST = conf["cost"]

        self.pool = self.MRC

    def growth(self, N):
        return self.RG * N * (1 - N / self.MRC)

    def harvest(self, x, N):
        return self.BETA * (x ** self.ALPHA) \
            * (N ** (1 - self.ALPHA))

    def reward(self, delta_n, pis):
        rewards = []
        # _lambda -> sustainability_goal
        _lambda = simple_step_function(delta_n)
        # xi -> wealth_goal
        for pi in pis:
            xi = simple_step_function(pi)
            rewards.append(self.W * _lambda + (1 - self.W) * xi)

        return rewards

    def step(self, xs):
        # Change game status
        X = sum(xs)
        harvest_level = self.harvest(X, self.pool)
        delta_n = self.growth_func(self.pool) - harvest_level
        self.pool += delta_n
        game_is_done = True if self.pool <= 5.0 else False
        
        # get feedback from env
        pis = [x / float(X) * harvest_level - self.COST * x \
            for x in xs]
        PI = sum(pis)

        # observations
        obs = [[x, pi, X, PI] for x, pi in zip(xs, pis)]

        return obs, self.reward(delta_n, pis), game_is_done

    def reset(self):
        self.pool = self.MRC


def simple_step_function(X):
    if X > 0:
        return 1
    elif X == 0:
        return 0
    else:
        return -1
