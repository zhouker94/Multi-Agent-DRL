import os


def build_argument_parser(parser):
    parser.add_argument('--model',
                        choices=["DQN", "DDPG"],
                        default="DQN",
                        help='RL algorithm')
    parser.add_argument('--n_agent',
                        type=int,
                        default=5,
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


def save_result(result_dict, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for r in result_dict:
        with open(os.path.join(save_path, r), "w+") as f:
            for i in result_dict[r]:
                f.write(str(i) + '\n')

    print("Results saved in path: {}".format(save_path))
