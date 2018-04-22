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
import environment


def main(argv):

    # initialize
    env = environment.GameEnv()
    print("weight is", const.WEIGHT)
    const.initialize(3, 2)
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
                if efforts[index] < const.MIN_EFFORT:
                    efforts[index] = const.MIN_EFFORT
                elif efforts[index] > const.MAX_EFFORT:
                    efforts[index] = const.MAX_EFFORT
                actions[index] = action

            resource, rewards, done = env.step(efforts)
            score += sum(rewards)
            next_state = np.reshape([resource], [1, const.STATE_SPACE])
            [player.store_experience(state, actions[index], rewards[index], next_state, done)
             for index, player in enumerate(players)]
            state = next_state

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
        player.sess.close()


if __name__ == '__main__':
    main(sys.argv)
