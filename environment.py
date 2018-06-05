#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/22 10:16
# @Author  : Hanwei Zhu
# @File    : environment.py

import constants as const

RESOURCE_CAPACITY_N_MAX = 1000.0
RESOURCE_CAPACITY_INIT = 1000.0
REPLENISHMENT_RATE = 0.5
ALPHA = 0.35
BETA = 0.4
COST_C = 0.5


class GameEnv(object):
    def __init__(self):
        print("Game start!")
        self.common_resource_pool = RESOURCE_CAPACITY_INIT

    @staticmethod
    def growth_func(n):
        rg = REPLENISHMENT_RATE
        return rg * n * (1 - n / RESOURCE_CAPACITY_N_MAX)

    @staticmethod
    def harvest_func(effort, n):
        return BETA * (effort ** ALPHA) * (n ** (1 - ALPHA))

    @staticmethod
    def reward_func(delta_n, pi_list):
        rewards = []

        if delta_n > 0:
            sustainability_goal = 1
        elif delta_n == 0:
            sustainability_goal = 0
        else:
            sustainability_goal = -1

        for pi in pi_list:
            if pi > 0:
                wealth_goal = 1
            elif pi == 0:
                wealth_goal = 0
            else:
                wealth_goal = -1
            r = const.WEIGHT * sustainability_goal + (1 - const.WEIGHT) * wealth_goal
            rewards.append(r)

        return rewards

    def step(self, efforts):
        effort_sum = sum(efforts)

        # change environment
        harvest_level = self.harvest_func(effort_sum, self.common_resource_pool)
        delta_n = int(self.growth_func(self.common_resource_pool) - harvest_level)
        self.common_resource_pool += delta_n

        # get feedback from env
        pi_list = list(map(lambda x: x / effort_sum * harvest_level - COST_C * x, efforts))
        game_is_done = False
        if self.common_resource_pool <= 0:
            game_is_done = True

        return (effort_sum, sum(pi_list)), self.reward_func(delta_n, pi_list), game_is_done

    def reset(self):
        self.common_resource_pool = const.RESOURCE_CAPACITY_INIT
        return self.common_resource_pool, 0, 0
