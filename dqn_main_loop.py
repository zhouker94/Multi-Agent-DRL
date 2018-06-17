# -*- coding: utf-8 -*-
# @Time    : 2018/3/5 下午1:53
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : dqn_main_loop.py
# @Software: PyCharm Community Edition


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import environment
from agents import agent
import argparse
import json


def main():

    # -------------- parameters initialize --------------

    env = environment.GameEnv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int)
    parser.add_argument('--sustainable_weight', type=float)
    parsed_args = parser.parse_args()

    conf = json.load(open('config.json', 'r'))
    env_conf = conf["game_config"]
    env_conf["sustain_weight"] = parsed_args.sustainable_weight
    env_conf["num_agents"] = parsed_args.n_agents

    dir_conf, opt = conf["dir_config"], conf["dqn"]

    agent_list = [agent.DqnAgent("DQN_" + str(i), opt)
                  for i in range(env_conf["num_agents"])]

    # -------------- start training --------------

    for a in agent_list:
        a.start(dir_path=dir_conf["model_save_path"])

    avg_scores = []
    global_step = 0
    for epoch in range(env_conf["train_epochs"]):

        if agent_list[0].epsilon <= opt["min_epsilon"]:
            break

        # state -> [X, Pi, x, pi]
        state = np.asarray([env.reset()])
        state = np.reshape(state, [1, opt["state_space"]])

        # actions -> [Increase effort, Decrease effort, IDLE]
        actions = [0] * env_conf["num_agents"]
        efforts = [env_conf["total_init_effort"] / env_conf["num_agents"]] * env_conf["num_agents"]
        score = 0

        for time in range(env_conf["max_round"]):
            for index, player in enumerate(agent_list):
                action = player.choose_action(state)
                actions[index] = action

                # increase
                if action == 0:
                    efforts[index] += env_conf["min_increment"]
                # decrease
                elif action == 1:
                    efforts[index] -= env_conf["min_increment"]

            resource, rewards, done = env.step(efforts)
            score += sum(rewards)
            next_state = np.reshape([resource], [1, opt["state_space"]])
            [player.save_transition(state, actions[index], rewards[index], next_state)
             for index, player in enumerate(agent_list)]
            state = next_state

            if done:
                break

            global_step += 1
            if not global_step % 100:
                [player.learn(global_step) for player in agent_list]

        score /= env_conf["num_agents"]
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(epoch, env_conf["train_epochs"], score, agent_list[0].epsilon))
        avg_scores.append(score)

    for a in agent_list:
        a.save(dir_path=dir_conf["model_save_path"])
        a.sess.close()

    plt.switch_backend('agg')
    plt.plot(avg_scores)
    plt.interactive(False)
    plt.xlabel('Epoch')
    plt.ylabel('Avg score')
    plt.savefig(dir_conf["log_path"] + 'training_plot')


if __name__ == '__main__':
    main()
