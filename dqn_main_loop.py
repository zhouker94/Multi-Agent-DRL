# -*- coding: utf-8 -*-
# @Time    : 2018/3/5 下午1:53
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : dqn_main_loop.py
# @Software: PyCharm Community Edition


import sys
import matplotlib.pyplot as plt
import numpy as np
import constants as const
import environment
from agents import agent
import argparse


def main(argv):

    env = environment.GameEnv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int)
    parser.add_argument('--sustainable_weight', type=float)
    parsed_args = parser.parse_args()

    # state -> [X, Pi, x, pi]
    const.initialize(state_space=4, action_space=3,
                     n_agents=parsed_args.n_agents,
                     weight=parsed_args.sustainable_weight)
    agent_list = []
    for i in range(const.N_AGENTS):
        agent_list.append(agent.DqnAgent("DQN_" + str(i)))

    avg_scores = []

    for epoch in range(const.TRAINING_EPISODES):
        if agent_list[0].epsilon <= const.EPSILON_MIN:
            break

        state = np.asarray([env.reset()])
        state = np.reshape(state, [1, const.STATE_SPACE])

        # actions -> [Increase effort, Decrease effort, IDLE]
        actions = [0] * const.N_AGENTS
        efforts = [const.INIT_EFFORT] * const.N_AGENTS
        score = 0
        transition = [dict() for i in range(len(agent_list))]
        copy_step = 0

        for time in range(const.MAX_STEP):
            for index, player in enumerate(agent_list):
                action = player.choose_action(state)
                actions[index] = action
                
                # increase
                if action == 0:
                    efforts[index] += const.MIN_INCREMENT
                # decrease
                elif action == 1:
                    efforts[index] -= const.MIN_INCREMENT

            resource, rewards, done = env.step(efforts)
            score += sum(rewards)
            next_state = np.reshape([resource], [1, const.STATE_SPACE])
            [player.store_experience(state, actions[index], rewards[index], next_state, done)
             for index, player in enumerate(agent_list)]
            state = next_state

            if done:
                break

        score /= const.N_AGENTS
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(epoch, const.TRAINING_EPISODES, score, agent_list[0].epsilon))
        avg_scores.append(score)

        if len(agent_list[0].memory) > const.MINI_BATCH_SIZE:
            for player in agent_list:
                player.learn()

        # copy weights from online q network to target q network
        if copy_step >= const.COPY_STEP:
            print("Copy start!")
            [player.update_target_q() for player in agent_list]
            print("Copy finished!")
            copy_step = 0
        else:
            copy_step += 1

    for a in agent_list:
        a.save_model()
        a.sess.close()

    plt.switch_backend('agg')
    plt.plot(avg_scores)
    plt.interactive(False)
    plt.xlabel('Epoch')
    plt.ylabel('Avg score')
    plt.savefig(const.LOG_PATH + 'training_plot')


if __name__ == '__main__':
    main(sys.argv)
