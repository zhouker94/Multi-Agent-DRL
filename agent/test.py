# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 下午8:47
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : cart_pole.py
# @Software: PyCharm Community Edition

import gym
import sys
import matplotlib.pyplot as plt
import agent
import numpy as np
import constants as const


def main(agrv):
    # initialize
    env = gym.make(const.ENV_NAME)
    env = env.unwrapped
    # const.STATE_SPACE = env.observation_space.shape[0]
    # const.ACTION_SPACE = env.action_space.n

    const.initialize(env.observation_space.shape[0], env.action_space.n)

    dqn_agent = agent.DqnAgent('DQNmodel')
    scores = []
    copy_step = 0

    for e in range(const.TRAINING_EPISODES):
        # env.render()
        state = env.reset()
        state = np.reshape(state, [1, const.STATE_SPACE])

        for time in range(const.MAX_STEP):
            action = dqn_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            x, x_dot, theta, theta_dot = next_state

            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            next_state = np.reshape(next_state, [1, const.STATE_SPACE])
            dqn_agent.store_experience(state, action, reward, next_state, done)
            state = next_state

            if copy_step >= const.COPY_STEP:
                dqn_agent.update_target_q()
                copy_step = 0
            else:
                copy_step += 1

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, const.TRAINING_EPISODES, time, dqn_agent.epsilon))
                scores.append(time)
                break

        if len(dqn_agent.memory) > const.MINI_BATCH_SIZE:
            dqn_agent.learn()

    dqn_agent.save_model()
    plt.plot(scores)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
