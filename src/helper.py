import os


def save_result(result_dict, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for r in result_dict:
        with open(os.path.join(save_path, r), "w+") as f:
            for i in result_dict[r]:
                f.write(str(i) + '\n')

    print("Results saved in path: {}".format(save_path))
