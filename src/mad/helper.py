import json
import os
from pathlib import Path
from typing import Dict


def save_experiment_result(result_dict: Dict[str, list[int | float]], save_path: Path):
    """Save experiment results to files in the specified directory.

    Parameters:
        result_dict (Dict[str, list[int | float]]): A dictionary where keys are filenames and values are lists
            of numbers to be saved in the corresponding files.
        save_path (Path): The directory where the files will be saved.

    Returns:
        None
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename, content in result_dict.items():
        with open(save_path.joinpath(filename), "w+") as f:
            for i in content:
                f.write(str(i) + '\n')

    print("Results saved in path: {}".format(save_path))


def save_experiment_metadata(agent_model: str, num_agent: int, sustainable_weight: float, save_path: Path):
    """Save experiment metadata to a JSON file in the specified directory.

    Parameters:
        agent_model (str): The learning model used for the experiment.
        num_agent (int): The number of agents in the experiment.
        sustainable_weight (float): The sustainable weight parameter used in the experiment.
        save_path (Path): The directory where the metadata file will be saved.

    Returns:
        None
    """
    with open(save_path.joinpath("metadata.json"), 'w') as f:
        json.dump(
            {
                "model": agent_model,
                "num_agent": num_agent,
                "sustainable_weight": sustainable_weight
            },
            f
        )
