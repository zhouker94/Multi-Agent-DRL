import argparse
import json
import os

import cpr_game
from agent import Agent
import helper


def main():
    parser = argparse.ArgumentParser()
    parsed_args = helper.build_argument_parser(parser)
    with open('config.json', 'r') as f:
        conf = json.load(f)

    MODEL_NAME = parsed_args.model
    N_AGENT = parsed_args.n_agent
    W = conf["game"]["sustainable_weight"] = \
        parsed_args.sustainable_weight
    LEARN_MODE = parsed_args.learn_mode
    CURRENT_VERSION = parsed_args.version

    if MODEL_NAME not in conf["model"]:
        raise NotImplementedError

    env_conf = conf["env"]
    SAVE_RESULT_PATH = os.path.join(
        env_conf["log_path"],
        "{}_model".format(MODEL_NAME),
        CURRENT_VERSION,
        "{}_{}".format(W, N_AGENT)
    )
    SAVE_MODEL_PATH = os.path.join(
        SAVE_RESULT_PATH,
        "checkpoints"
    )

    # Init game environment
    game = cpr_game.CPRGame(conf["game"])
    # Init agents
    agent_list = [
        Agent(
            "agent_{}".format(i),
            conf["model"][MODEL_NAME],
        )
        for i in range(N_AGENT)
    ]

    if LEARN_MODE == "train":
        avg_scores = []
        epsilon = env_conf["init_epsilon"]
        state_list = [[0] * env_conf["state_space"]] * N_AGENT
        next_state_list = [[0] * env_conf["state_space"]] * N_AGENT
        reward_list = [0] * N_AGENT
        epoch = 0
        while epsilon >= env_conf["min_epsilon"]:
            # Reset Game Environment
            game.reset()
            effort_list = [env_conf["total_init_effort"] / N_AGENT] * N_AGENT
            score = 0.0
            for _ in range(env_conf["max_train_round"]):
                for i, agent in enumerate(agent_list):

                    if MODEL_NAME == "DDPG":
                        action = agent.act(
                            state_list[i],
                            epsilon=epsilon,
                            upper_bound=game.pool / N_AGENT
                        )
                    elif MODEL_NAME == "DQN":
                        action = agent.act(
                            state_list[i],
                            epsilon=epsilon,
                            pre_action=effort_list[i]
                        )

                    effort_list[i] = action
                    agent.remember(
                        state_list[i],
                        action,
                        reward_list[i],
                        next_state_list[i]
                    )
                    effort_list[i] = action

                next_state_list, reward_list, done = game.step(effort_list)
                score += sum(reward_list) / N_AGENT
                if done:
                    break

            for agent in agent_list:
                agent.learn()

            print("epoch: {}, score: {}, e: {:.2}"
                  .format(epoch, score, epsilon))
            epsilon -= env_conf["epsilon_decay"]
            epoch += 1
            avg_scores.append(score)

        for agent in agent_list:
            agent.save(SAVE_MODEL_PATH)

        helper.save_result(avg_scores, SAVE_RESULT_PATH)

    elif LEARN_MODE == "test":
        pass


if __name__ == "__main__":
    main()
