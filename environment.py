#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/22 10:16
# @Author  : Hanwei Zhu
# @File    : environment.py


class GameEnv(object):
    def __init__(self, conf):
        self.conf = conf
        self._w = self.conf["sustain_weight"]
        print("Game start!")
        self.common_resource_pool = self.conf["resource_capacity_init"]

    def growth_func(self, n):
        rg = self.conf["replenishment_rate"]
        return rg * n * (1 - n / self.conf["resource_capacity_n_max"])

    def harvest_func(self, effort, n):
        return self.conf["beta"] * (effort ** self.conf["alpha"]) * (n ** (1 - self.conf["alpha"]))

    def reward_func(self, delta_n, pi_list):
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
            r = self._w * sustainability_goal + (1 - self._w) * wealth_goal
            rewards.append(r)

        return rewards

    def step(self, efforts):
        effort_sum = sum(efforts)

        # change environment
        harvest_level = self.harvest_func(effort_sum, self.common_resource_pool)
        delta_n = int(self.growth_func(self.common_resource_pool) - harvest_level)
        self.common_resource_pool += delta_n

        # get feedback from env
        pi_list = list(map(lambda x: x / effort_sum * harvest_level - self.conf["cost_c"] * x, efforts))
        game_is_done = False
        if self.common_resource_pool <= 0:
            game_is_done = True

        return [effort_sum, sum(pi_list)], self.reward_func(delta_n, pi_list), game_is_done

    def reset(self):
        self.common_resource_pool = self.conf["resource_capacity_init"]
        return [0.0, 0.0]
