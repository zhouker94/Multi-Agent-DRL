import argparse
import json
import os

import cpr_game
import agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        choices=["dqn", "ddpg"],
                        default="dqn",
                        help='RL algorithm')
    parser.add_argument('--n_agent',
                        type=int,
                        default=1,
                        help='The number of agents')
    parser.add_argument('--sustainable_weight',
                        type=float,
                        default=0.5,
                        help='Weight of sustainability')
    parser.add_argument('--learn_mode',
                        choices=["train", "test"],
                        default="train",
                        help='Train or test mode')
    parser.add_argument('--v',
                        type=str,
                        default="v_00",
                        help='Current model version')
    parsed_args = parser.parse_args()
    conf = json.load(open('config.json', 'r'))

    MODEL_NAME = parsed_args.model
    N_AGENT = parsed_args.n_agent
    W = conf["game"]["sustainable_weight"] = \
        parsed_args.sustainable_weight
    LEARN_MODE = parsed_args.learn_mode

    if not MODEL_NAME in conf["model"]:
        raise NotImplementedError

    env_conf = conf["env"]
    SAVE_MODEL_PATH = os.path.join(
        env_conf["log_path"],
        "model",
        "{}_{}".format(W, N_AGENT)
    )
    SAVE_RESULT_PATH = os.path.join(
        env_conf["log_path"],
        "result",
        "{}_{}".format(W, N_AGENT)
    )

    # Init game
    game = cpr_game.CPRGame(conf["game"])
    # Init agents
    agent_list = [
        agent.Agent("agent:{}".format(i),
                    conf["model"][MODEL_NAME],
                    LEARN_MODE,
                    SAVE_MODEL_PATH)
        for i in range(N_AGENT)
    ]

    global_step = 0
    if LEARN_MODE == "train":
        avg_scores = []

    elif LEARN_MODE == "test":
        pass


if __name__ == "__main__":
    main()
