import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import helper
from agent import Agent
from cpr_game import CPRGame


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        choices=["DQN", "DDPG"],
        default="DDPG",
        help='Reinforcement Learning Algorithms'
    )

    parser.add_argument(
        '--n_agent',
        type=int,
        default=5,
        help='The number of agents'
    )

    parser.add_argument(
        '--sustainable_weight',
        type=float,
        default=0.5,
        help='Weight of sustainability goal'
    )

    parser.add_argument(
        '--run_mode',
        choices=["train", "test"],
        default="train",
        help='Train or test mode'
    )

    args = parser.parse_args()

    with open('config/config.json', 'r') as f:
        conf = json.load(f)

    conf["hyper_parameter"]["sustainable_weight"] = args.sustainable_weight

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    if args.model not in conf["model"]:
        raise NotImplementedError("Model Type Not Support")

    env_conf = conf["env"]
    save_result_path = Path(env_conf["log_path"]). \
        joinpath(f"{args.model}_model_{timestamp}_{args.sustainable_weight}_{args.n_agent}")

    save_model_path = os.path.join(
        save_result_path,
        "checkpoints"
    )

    # Init game environment
    cpr = CPRGame(conf["hyper_parameter"])
    states = np.zeros((args.n_agent, env_conf["state_space"]))
    next_states = np.zeros((args.n_agent, env_conf["state_space"]))
    rewards = np.zeros(args.n_agent)

    # -------------- train mode --------------

    if args.run_mode == "train":
        # Init agents
        agents = list(map(lambda _: Agent(conf["model"][args.model]),
                          range(args.n_agent)))

        avg_scores = []
        epsilon = env_conf["init_epsilon"]
        epoch = 0
        while epsilon >= env_conf["min_epsilon"]:
            # Reset Game Environment
            cpr.reset()
            efforts = np.array([env_conf["total_init_effort"] / args.n_agent] * args.n_agent)
            score = 0.0
            for _ in range(env_conf["max_train_round"]):
                for i, agent in enumerate(agents):

                    if args.model == "DDPG":
                        action = agent.act(
                            states[i],
                            epsilon=epsilon,
                            upper_bound=cpr.pool / args.n_agent
                        )
                    else:
                        action = agent.act(
                            states[i],
                            epsilon=epsilon,
                            pre_action=effort_list[i]
                        )

                    efforts[i] = action
                    agent.remember(
                        states[i],
                        action,
                        rewards[i],
                        next_states[i]
                    )

                next_states, rewards, done = cpr.step(efforts)
                score += sum(rewards) / args.n_agent
                if done:
                    break

            for agent in agents:
                agent.learn()

            print(f"epoch: {epoch}, score: {score}, e: {epsilon:.2}")
            epsilon -= env_conf["epsilon_decay"]
            epoch += 1
            avg_scores.append(score)

        for agent in agents:
            agent.close(save_model_path)

        helper.save_result(
            {"train_avg_score.txt": avg_scores},
            save_result_path
        )

    # -------------- test mode --------------

    elif args.run_mode == "test":
        # Init agents
        agent_list = [
            Agent(
                "agent_{}".format(i),
                conf["model"][args.model],
                save_model_path
            )
            for i in range(args.n_agent)
        ]

        avg_asset_seq = [0]
        pool_level_seq = []
        avg_score_seq = []

        for t in range(env_conf["max_test_round"]):
            pool_level_seq.append(cpr.pool)
            effort_list = [env_conf["total_init_effort"] / args.n_agent] * args.n_agent
            for i, agent in enumerate(agent_list):
                if args.model == "DDPG":
                    action = agent.act(
                        states[i],
                        upper_bound=cpr.pool / args.n_agent
                    )
                elif args.model == "DQN":
                    action = agent.act(
                        states[i],
                        pre_action=effort_list[i]
                    )

                effort_list[i] = action
                agent.remember(
                    states[i],
                    action,
                    rewards[i],
                    next_states[i]
                )
                effort_list[i] = action

            next_states, rewards, done = cpr.step(effort_list)

            avg_score_seq.append(sum(rewards) / args.n_agent)
            avg_asset_seq.append(
                avg_asset_seq[-1] + next_states[0][3] / args.n_agent)

            if done:
                break

        for agent in agent_list:
            agent.close()

        helper.save_result(
            {
                "test_avg_score.txt": avg_asset_seq,
                "test_assets.txt": avg_asset_seq,
                "test_resource_level.txt": pool_level_seq
            },
            save_result_path
        )


if __name__ == "__main__":
    main()
