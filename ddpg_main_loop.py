# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 下午9:04
# @Author  : Hanwei Zhu
# @Email   : hanweiz@student.unimelb.edu.au
# @File    : ddpg_main_loop.py
# @Software: PyCharm Community Edition


import matplotlib.pyplot as plt
import numpy as np
import environment
from agents import ddpg_agent
import argparse
import json


def main():
    # -------------- parameters initialize --------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=1)
    parser.add_argument('--sustainable_weight', type=float, default=0.5)
    parser.add_argument('--is_test', type=bool, default=False)
    parsed_args = parser.parse_args()

    conf = json.load(open('config.json', 'r'))
    training_conf = conf["training_config"]

    env_conf = conf["env_config"]
    env_conf["sustain_weight"] = parsed_args.sustainable_weight
    env_conf["num_agents"] = parsed_args.n_agents
    env = environment.GameEnv(env_conf)

    dir_conf, opt = conf["dir_config"], conf["ddpg"]
    dir_conf["model_save_path"] = dir_conf["model_save_path"] + '_' + \
                                  str(env_conf["sustain_weight"]) + '_' + \
                                  str(training_conf["num_agents"]) + '/'

    avg_scores = []
    global_step = 0

    # -------------- start train mode --------------
    if not parsed_args.is_test:

        agent_list = []
        for i in range(training_conf["num_agents"]):
            player = ddpg_agent.DDPGAgent("DDPG_" + str(i), opt)
            player.start(dir_path=dir_conf["model_save_path"])
            agent_list.append(player)

        for epoch in range(training_conf["train_epochs"]):
            if agent_list[0].epsilon <= opt["min_epsilon"]:
                break

            # state -> [X, Pi]
            state = env.reset()

            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf["num_agents"]
            score = 0

            for time in range(training_conf["max_round"]):
                # actions -> [Increase effort, Decrease effort, IDLE]
                actions = [0] * training_conf["num_agents"]

                for index, player in enumerate(agent_list):
                    action = player.choose_action(np.expand_dims(state, axis=0))
                    actions[index] = action

                    # increase
                    if action == 0:
                        efforts[index] += training_conf["min_increment"]
                    # decrease
                    elif action == 1:
                        efforts[index] -= training_conf["min_increment"]

                    if efforts[index] <= 1:
                        efforts[index] = 1

                next_state, rewards, done = env.step(efforts)
                score += sum(rewards)

                [player.save_transition(state, actions[index], rewards[index])
                 for index, player in enumerate(agent_list)]
                state = next_state

                global_step += 1

                if done:
                    break

            if not epoch % 2:
                [player.learn(global_step) for player in agent_list]

            score /= training_conf["num_agents"]

            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(epoch, env_conf["train_epochs"], score, agent_list[0].epsilon))

            avg_scores.append(score)

        for a in agent_list:
            a.save(dir_path=dir_conf["model_save_path"])
            a.sess.close()

        # -------------- save results --------------
        plt.switch_backend('agg')
        plt.plot(avg_scores)
        plt.interactive(False)
        plt.xlabel('Epoch')
        plt.ylabel('Avg score')
        plt.savefig(dir_conf["model_save_path"] + 'ddpg_training_plot')

        with open(dir_conf["model_save_path"] + 'ddpg_avg_score.txt', "w+") as f:
            for r in avg_scores:
                f.write(str(r) + '\n')

    # -------------- start test mode --------------
    else:
        agent_list = []
        for i in range(training_conf["num_agents"]):
            player = ddpg_agent.DDPGAgent("DDPG_" + str(i), opt, learning_mode=False)
            player.start(dir_path=dir_conf["model_save_path"])
            agent_list.append(player)

        resource_level = []
        for epoch in range(1):
            # state -> [X, Pi]
            state = env.reset()

            efforts = [training_conf["total_init_effort"] / training_conf["num_agents"]] * training_conf["num_agents"]
            score = 0

            for time in range(training_conf["max_round"]):
                resource_level.append(env.common_resource_pool)

                # actions -> [Increase effort, Decrease effort, IDLE]
                actions = [0] * training_conf["num_agents"]

                for index, player in enumerate(agent_list):
                    action = player.choose_action(np.expand_dims(state, axis=0))
                    actions[index] = action

                    # increase
                    if action == 0:
                        efforts[index] += training_conf["min_increment"]
                    # decrease
                    elif action == 1:
                        efforts[index] -= training_conf["min_increment"]

                    if efforts[index] <= 1:
                        efforts[index] = 1

                next_state, rewards, done = env.step(efforts)
                score = sum(rewards)
                score /= training_conf["num_agents"]
                avg_scores.append(score)
                state = next_state

                global_step += 1

                if done:
                    break

            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(epoch, training_conf["test_epochs"], score, agent_list[0].epsilon))

        for a in agent_list:
            a.sess.close()

        # -------------- save results --------------
        plt.switch_backend('agg')
        plt.plot(avg_scores)
        plt.interactive(False)
        plt.xlabel('Epoch')
        plt.ylabel('Avg score')
        plt.savefig(dir_conf["model_save_path"] + 'ddpg_test_plot')

        with open(dir_conf["model_save_path"] + 'ddpg_test_avg_score.txt', "w+") as f:
            for r in avg_scores:
                f.write(str(r) + '\n')

        with open(dir_conf["model_save_path"] + "ddpg_test_resource_level.txt", "w+") as f:
            for r in resource_level:
                f.write(str(r) + '\n')


if __name__ == '__main__':
    main()
