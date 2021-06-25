import argparse
from pathlib import Path

import numpy as np

from agent import Agent
from cpr_env import CPREnvironment


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        choices=["DQN", "DDPG"],
                        default="DQN",
                        help='RL algorithm')
    parser.add_argument('--num_player',
                        type=int,
                        default=5,
                        help='The number of players')
    parser.add_argument('--sustainable_weight',
                        type=int,
                        default=5,
                        help='Sustainable goal')

    args = parser.parse_args()

    model_type = args.model
    num_player = args.num_player

    cpr = CPREnvironment(num_player=num_player, s_weight=args.sustainable_weight)
    save_path = Path("./log").joinpath(f"{model_type}")

    # Init game environment
    # Init
    agents = [Agent(i, "DDPG") for i in range(num_player)]

    avg_scores = []
    epsilon = env_conf["init_epsilon"]
    epoch = 0
    while epsilon >= env_conf["min_epsilon"]:
        # Reset Game Environment
        game.reset()
        efforts = np.array([env_conf["total_init_effort"] / N_AGENT] * N_AGENT)
        score = 0.0
        for _ in range(env_conf["max_train_round"]):
            for i, agent in enumerate(agents):

                if MODEL_NAME == "DDPG":
                    action = agent.act(
                        states[i],
                        epsilon=epsilon,
                        upper_bound=game.pool / N_AGENT
                    )
                elif MODEL_NAME == "DQN":
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

            next_states, rewards, done = game.step(efforts)
            score += sum(rewards) / N_AGENT
            if done:
                break

        [agent.learn() for agent in agents]

        print("epoch: {}, score: {}, e: {:.2}"
              .format(epoch, score, epsilon))
        epsilon -= env_conf["epsilon_decay"]
        epoch += 1
        avg_scores.append(score)

    [agent.close(SAVE_MODEL_PATH) for agent in agents]
    helper.save_result(
        {"train_avg_score.txt": avg_scores},
        SAVE_RESULT_PATH
    )


if __name__ == "__main__":
    main()
