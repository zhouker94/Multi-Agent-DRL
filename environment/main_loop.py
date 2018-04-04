# -*- coding: utf-8 -*-
# @Time    : 2018/3/5 下午1:53
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : main_loop.py
# @Software: PyCharm Community Edition

import sys
import constants as const
import functions as func
import agent


class GameEnv(object):
    def __init__(self):
        print("Game start!")
        self.common_resource_pool = const.RESOURCE_CAPACITY_N_MAX

    def step(self, harvest_level):
        self.common_resource_pool = self.common_resource_pool + \
                                    func.growth_function(self.common_resource_pool) - \
                                    harvest_level
        done = True if self.common_resource_pool <= 0 else False
        return self.common_resource_pool, done


def main(argv):
    # initialize

    env = GameEnv()
    resource = env.common_resource_pool

    curr_turn = 0
    while curr_turn <= const.MAX_TURNS:
        # agents make decision
        harvest_level = 0

        players = []
        # TODO: plugin my agent
        for player in range(const.N_AGENTS):
            players.append(agent.DqnAgent('DQNmodel'+str(player)))
            effort = int(input("player" + str(player) + "input"))
            harvest_level += func.harvest_function(effort, resource)

        resource, done = env.step(harvest_level)

        print(curr_turn, ": current resource is", resource)
        if done:
            print('finished!', resource, curr_turn)

        curr_turn += 1


if __name__ == '__main__':
    main(sys.argv)
