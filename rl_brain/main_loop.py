# -*- coding: utf-8 -*-
# @Time    : 2018/3/5 下午1:53
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : main_loop.py
# @Software: PyCharm Community Edition

import sys
import numpy as np
import agent
import constants as const
import matplotlib.pyplot as plt


def harvest_function(effort, n):
    return const.BETA * (effort ** const.ALPHA) * (n ** (1 - const.ALPHA))


def growth_function(n):
    rg = const.REPLENISHMENT_RATE
    return rg * n * (1 - n / const.RESOURCE_CAPACITY_N_MAX)


class GameEnv(object):
    def __init__(self):
        print("Game start!")
        self.common_resource_pool = const.RESOURCE_CAPACITY_N_MAX * 1.0

    def step(self, harvest_level):
        self.common_resource_pool = self.common_resource_pool + \
                                    growth_function(self.common_resource_pool) - \
                                    harvest_level
        # print('see!', self.common_resource_pool)
        done = True if self.common_resource_pool <= 0 else False
        return self.common_resource_pool, done

    def reset(self):
        self.common_resource_pool = const.RESOURCE_CAPACITY_N_MAX * 1.0
        return self.common_resource_pool


def main(argv):
    # initialize

    env = GameEnv()
    const.initialize(1, 2)
    copy_step = 0
    scores = []

    players = []
    for player in range(const.N_AGENTS):
        # print(player)
        players.append(agent.DqnAgent(str(player)))

    for e in range(const.TRAINING_EPISODES):

        if players[0].epsilon <= const.EPSILON_MIN:
            break

        env.reset()
        state = np.asarray([env.reset()])
        state = np.reshape(state, [1, const.STATE_SPACE])

        actions = [0] * const.N_AGENTS
        efforts = [const.INIT_EFFORT] * const.N_AGENTS
        harvests = [0] * const.N_AGENTS

        for time in range(const.MAX_STEP):
            score = 0
            for index, player in enumerate(players):
                action = player.choose_action(state)
                if action == 0:
                    efforts[index] += const.MIN_INCREMENT
                else:
                    efforts[index] -= const.MIN_INCREMENT
                if efforts[index] <= 0:
                    efforts[index] = 1
                harvests[index] = harvest_function(efforts[index], env.common_resource_pool)
                actions[index] = action

            score += sum(harvests)
            resource, done = env.step(score)

            if done:
                harvests = [-100] * const.N_AGENTS

            next_state = np.reshape([resource], [1, const.STATE_SPACE])
            [player.store_experience(state, actions[index], harvests[index], next_state, done)
             for index, player in enumerate(players)]

            state = next_state

            # copy weights from online q network to target q network
            if copy_step >= const.COPY_STEP:
                [player.update_target_q() for player in players]
                copy_step = 0
            else:
                copy_step += 1

            '''
            print("efforts", efforts)
            print("harvests", harvests)
            print("remain:", env.common_resource_pool)
            '''

            if done:
                score = score - 100 / const.N_AGENTS
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, const.TRAINING_EPISODES, score, players[0].epsilon))
                scores.append(score)
                break

        if len(players[0].memory) > const.MINI_BATCH_SIZE:
            [player.learn() for player in players]

    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
