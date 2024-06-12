import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from . import helper
from .agent import Agent
from .cpr_environment import CPREnvironment

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_commandline_arguments() -> argparse.Namespace:
    """This function parse use input arguments from command line

    Returns: parsed arguments from command line

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        choices=["DQN", "DDPG"],
        default="DDPG",
        help='Reinforcement Learning Algorithms'
    )

    parser.add_argument(
        '--num_agent',
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

    return parser.parse_args()


def get_result_path() -> Path:
    """This function gets result filepath on your local file system

    If the filepath doesn't exist, this function will create one for you

    Returns: result path

    """
    current_time = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    return Path("./out").joinpath(current_time)


def get_checkpoint_path(result_filepath: Path) -> Path:
    """Generate the path to the checkpoints directory based on the provided result filepath.

    If the filepath doesn't exist, this function will create one for you

    Parameters:
        result_filepath (Path): A Path object representing the filepath where the result is stored.

    Returns:
        Path: A Path object representing the path to the checkpoints directory relative to the result_filepath.
    """
    return result_filepath.joinpath("checkpoints")


def main():
    args = parse_commandline_arguments()

    num_agents: int = args.num_agent
    agent_model: str = args.model
    sustainable_weight: float = args.sustainable_weight

    with open('config/config.json', 'r') as f:
        conf = json.load(f)

    conf["hyper_parameter"]["sustainable_weight"] = args.sustainable_weight

    env_conf = conf["env"]
    result_path = get_result_path()
    checkpoint_path = get_checkpoint_path(result_path)

    # Init game environment
    cpr = CPREnvironment(conf["hyper_parameter"])
    states = np.zeros((num_agents, env_conf["state_space"]))
    next_states = np.zeros((num_agents, env_conf["state_space"]))
    rewards = np.zeros(num_agents)

    # -------------- train mode --------------

    if args.run_mode == "train":
        # Setup
        agents = []
        for _ in range(num_agents):
            agents.append(Agent(conf["model"][args.model]))

        scores = []
        epsilon = env_conf["init_epsilon"]
        epoch = 0

        while epsilon >= env_conf["min_epsilon"]:
            # Reset Game Environment
            cpr.reset()
            efforts = np.array([env_conf["total_init_effort"] / num_agents] * num_agents)
            score = 0.0
            for _ in range(env_conf["max_train_round"]):
                for i, agent in enumerate(agents):

                    if args.model == "DDPG":
                        action = agent.act(
                            states[i],
                            epsilon=epsilon,
                            upper_bound=cpr.pool / num_agents
                        )
                    else:
                        action = agent.act(
                            states[i],
                            epsilon=epsilon,
                            pre_action=efforts.tolist()[i].tolist()
                        )

                    efforts[i] = action
                    agent.remember(
                        states[i],
                        action,
                        rewards[i],
                        next_states[i]
                    )

                next_states, rewards, done = cpr.step(efforts)
                score += sum(rewards) / num_agents
                if done:
                    break

            for agent in agents:
                agent.learn()

            logger.info(f"epoch: {epoch}, score: {score}, e: {epsilon:.2}")
            epsilon -= env_conf["epsilon_decay"]
            epoch += 1
            scores.append(score)

        for agent in agents:
            agent.close(checkpoint_path)

        helper.save_experiment_result(
            {"train_avg_score.txt": scores},
            result_path
        )
        helper.save_experiment_metadata(agent_model, num_agents, sustainable_weight, result_path)

    # -------------- test mode --------------

    elif args.run_mode == "test":
        # Init agents
        agent_list = [
            Agent(
                "agent_{}".format(i),
                conf["model"][args.model],
                checkpoint_path
            )
            for i in range(num_agents)
        ]

        avg_asset_seq = [0]
        pool_level_seq = []
        avg_score_seq = []

        for t in range(env_conf["max_test_round"]):
            pool_level_seq.append(cpr.pool)
            effort_list = [env_conf["total_init_effort"] / num_agents] * num_agents
            for i, agent in enumerate(agent_list):
                if args.model == "DDPG":
                    action = agent.act(
                        states[i],
                        upper_bound=cpr.pool / num_agents
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

            avg_score_seq.append(sum(rewards) / num_agents)
            avg_asset_seq.append(
                avg_asset_seq[-1] + next_states[0][3] / num_agents)

            if done:
                break

        for agent in agent_list:
            agent.close()

        helper.save_experiment_result(
            {
                "test_avg_score.txt": avg_asset_seq,
                "test_assets.txt": avg_asset_seq,
                "test_resource_level.txt": pool_level_seq
            },
            result_path
        )
        helper.save_experiment_metadata(agent_model, num_agents, sustainable_weight, result_path)


if __name__ == "__main__":
    main()
