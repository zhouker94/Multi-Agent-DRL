#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/25 16:33
# @Author  : Hanwei Zhu
# @File    : drqn_main_loop.py

import sys

import matplotlib.pyplot as plt
import numpy as np

import constants as const
import environment
from agents import agent


def main(argv):
    # initialize
    env = environment.GameEnv()
    print("weight is", const.WEIGHT)
    const.initialize(3, 2)
    copy_step = 0
    scores = []

    players = []
    for player in range(const.N_AGENTS):
        players.append(agent.DrqnAgent("DRQN_" + str(player)))

    for e in range(const.TRAINING_EPISODES):
        if players[0].epsilon <= const.EPSILON_MIN:
            break

        # shape: [Batch_size, Time step, State space]
        state = np.zeros((1, const.MAX_STEP, const.STATE_SPACE))
        state[0][0] = np.asarray([env.reset()])

        rewards = [[] for _ in range(const.N_AGENTS)]
        actions = [[] for _ in range(const.N_AGENTS)]
        terminate = []

        efforts = [const.INIT_EFFORT] * const.N_AGENTS
        score = 0

        for time in range(1, const.MAX_STEP):

            for index, player in enumerate(players):
                action = player.choose_action(state)

                if action == 0:
                    efforts[index] += const.MIN_INCREMENT
                else:
                    efforts[index] -= const.MIN_INCREMENT

                if efforts[index] < const.MIN_EFFORT:
                    efforts[index] = const.MIN_EFFORT
                elif efforts[index] > const.MAX_EFFORT:
                    efforts[index] = const.MAX_EFFORT

                actions[index].append(action)

            next_state, curr_rewards, done = env.step(efforts)
            score += sum(curr_rewards)

            for i, rw in enumerate(curr_rewards):
                rewards[i].append(rw)

            if done:
                terminate.append(True)
                break

            terminate.append(False)
            state[0][time] = np.reshape(next_state, [1, const.STATE_SPACE])

        score /= const.N_AGENTS
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e, const.TRAINING_EPISODES, score, players[0].epsilon))

        [player.store_experience(state, actions[index], rewards[index], terminate)
            for index, player in enumerate(players)]

        scores.append(score)

        if len(players[0].memory) > const.DRQN_MINI_BATCH_SIZE:
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

    for player in players:
        player.save_model()
        player.sess.close()

    plt.plot(scores)
    plt.interactive(False)
    plt.xlabel('Epoch')
    plt.ylabel('Avg score')
    plt.show()


if __name__ == '__main__':
    main(sys.argv)