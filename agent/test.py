# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 下午8:47
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : cart_pole.py
# @Software: PyCharm Community Edition

import gym
import sys

import agent
import numpy as np
import constants as const
import tensorflow as tf
import q_network
import os


def main(agrv):
    # initialize
    env = gym.make(const.ENV_NAME)
    env = env.unwrapped
    # const.STATE_SPACE = env.observation_space.shape[0]
    # const.ACTION_SPACE = env.action_space.n

    const.initialize(env.observation_space.shape[0], env.action_space.n)

    dqn_agent = agent.DqnAgent('DQNmodel')

    for e in range(const.TRAINING_EPISODES):
        # env.render()
        state = env.reset()
        state = np.reshape(state, [1, const.STATE_SPACE])
        
        C = 0

        for time in range(const.MAX_STEP):
            action = dqn_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, const.STATE_SPACE])
            # print(state, action, reward, next_state, done)
            dqn_agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            
            if C >= const.COPY_STEP:
                dqn_agent.update_target_q()
                C = 0
            C += 1

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, const.TRAINING_EPISODES, time, dqn_agent.epsilon))
                break

        if dqn_agent.replay_buffer.get_size() > const.MINI_BATCH_SIZE:
            dqn_agent.update_online_q()


if __name__ == '__main__':
    main(sys.argv)
