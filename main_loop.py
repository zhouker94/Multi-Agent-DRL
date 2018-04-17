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
        self.common_resource_pool = const.RESOURCE_CAPACITY_N_MAX

    def step(self, efforts):
        effort_sum = sum(efforts)
        harvest_level = harvest_function(effort_sum, self.common_resource_pool)
        delta_N = growth_function(self.common_resource_pool) - harvest_level
        self.common_resource_pool += delta_N
        
        # reward function
        pi_list = [x / effort_sum * harvest_level - 0.5 * x for x in efforts]
        if delta_N > 0:
            sustianability_goal = 1
        elif delta_N == 0:
            sustianability_goal = 0
        else:
            sustianability_goal = -1

        for pi in pi_list:
            if pi > 0:
                wealth_goal = 1
            elif pi == 0:
                wealth_goal = 0
            else:
                wealth_goal = -1

            r = 0.5 * sustianability_goal + 0.5 * wealth_goal
            rewards.append(r)
        

        if self.common_resource_pool <= 0:
            done = True
        else:
            done = False

        return self.common_resource_pool, rewards, done

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
        players.append(agent.DqnAgent("DQN_" + str(player)))

    for e in range(const.TRAINING_EPISODES):
        if players[0].epsilon <= const.EPSILON_MIN:
            break

        state = np.asarray([env.reset()])
        state = np.reshape(state, [1, const.STATE_SPACE])

        actions = [0] * const.N_AGENTS
        efforts = [const.INIT_EFFORT] * const.N_AGENTS
        score = 0

        for time in range(const.MAX_STEP):
            for index, player in enumerate(players):
                action = player.choose_action(state)
                if action == 0:
                    efforts[index] += const.MIN_INCREMENT
                else:
                    efforts[index] -= const.MIN_INCREMENT
                if efforts[index] <= 0:
                    efforts[index] = 1
                actions[index] = action

            resource, rewards, done = env.step(efforts)
            score += sum(rewards)
            next_state = np.reshape([resource], [1, const.STATE_SPACE])
            [player.store_experience(state, actions[index], rewards[index], next_state, done)
             for index, player in enumerate(players)]
            state = next_state

            '''
            print("efforts", efforts)
            print("reward", rewards)
            print("remain:", env.common_resource_pool)
            '''

            if done:
                break

        score /= const.N_AGENTS
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e, const.TRAINING_EPISODES, score, players[0].epsilon))
        scores.append(score)

        if len(players[0].memory) > const.MINI_BATCH_SIZE:
            for player in players:
                player.learn()

        # copy weights from online q network to target q network
        if copy_step >= const.COPY_STEP:
            print("Copy start!")
            [player.update_target_q() for player in players]
            print("Copy finished!")
            copy_step = 0
        else:
            copy_step += 1

    plt.plot(scores)
    plt.interactive(False)
    plt.xlabel('Epoch')
    plt.ylabel('Avg score')
    plt.show()

    for player in players:
        player.save_model()


if __name__ == '__main__':
    main(sys.argv)
