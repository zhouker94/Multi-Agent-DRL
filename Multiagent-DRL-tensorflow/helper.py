import os
import matplotlib.pyplot as plt


def build_argument_parser(parser):
    parser.add_argument('--model',
                        choices=["DQN", "DDPG"],
                        default="DDPG",
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
    parser.add_argument('--version',
                        type=str,
                        default="v_00",
                        help='Current model version')
    return parser.parse_args()


def save_result(avg_scores, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.switch_backend('agg')
    plt.plot(avg_scores)
    plt.interactive(False)
    plt.xlabel('Epoch')
    plt.ylabel('Avg score')
    plt.savefig(os.path.join(save_path, 'training_plot'))

    with open(os.path.join(save_path, 'train_avg_score.txt'), "w+") as f:
        for r in avg_scores:
            f.write(str(r) + '\n')

    print("Results saved in path: {}".format(save_path))
