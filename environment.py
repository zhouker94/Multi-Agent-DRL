#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/22 10:16
# @Author  : Hanwei Zhu
# @File    : environment.py


import constants as const


def harvest_function(effort, n):
    return const.BETA * (effort ** const.ALPHA) * (n ** (1 - const.ALPHA))


def growth_function(n):
    rg = const.REPLENISHMENT_RATE
    return rg * n * (1 - n / const.RESOURCE_CAPACITY_N_MAX)


class GameEnv(object):
    def __init__(self):
        print("Game start!")
        self.common_resource_pool = const.RESOURCE_CAPACITY_N_MAX

    def step(self, efforts):
        effort_sum = sum(efforts)
        harvest_level = harvest_function(effort_sum, self.common_resource_pool)
        delta_n = growth_function(self.common_resource_pool) - harvest_level
        self.common_resource_pool += delta_n
        rewards = []
        # reward function
        pi_list = [x / effort_sum * harvest_level - 0.5 * x for x in efforts]
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

            r = 0.5 * sustainability_goal + 0.5 * wealth_goal
            rewards.append(r)

        if self.common_resource_pool <= 0:
            done = True
            rewards = [-20] * const.N_AGENTS
        else:
            done = False

        return (self.common_resource_pool / 100, effort_sum, sum(pi_list)), rewards, done

    def reset(self):
        self.common_resource_pool = const.RESOURCE_CAPACITY_N_MAX * 1.0
        return self.common_resource_pool / 100, 0, 0
