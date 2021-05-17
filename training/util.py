import json
from numpy import ndarray

from typing import TextIO, Tuple, Dict, List

import math
import os

import numpy as np
from vizdoom import GameVariable, DoomGame


def get_input_shape(algorithm: str, img_width: int, img_height: int) -> Tuple:
    """
    Determine the shape of the input for the given algorithm and image dimensions
    :param algorithm: The DRL algorithm to determine the input shape for the model
    :param img_width: The number of pixels in the x-axis of the image
    :param img_height: The number of pixels in the y-axis of the image
    :return: The input shape
    """
    img_channels = 3  # RGB
    trace_length = 4  # Temporal Dimension
    if algorithm == 'DRQN':
        return trace_length, img_width, img_height, img_channels  # 4x64x64x3
    else:
        return img_width, img_height, trace_length  # 64x64x4


def next_model_version(path: str) -> int:
    version = 0
    while os.path.exists(path.replace('*', str(version))):
        version += 1
    return version


def next_model_path(path: str) -> str:
    """
    Get the path of the next model by replacing the placeholder in the model path with the next version
    :param path: Path of the model with a placeholder for the version
    :return: The path to the model with the next version
    """
    return path.replace('*', str(next_model_version(path)))


def latest_model_path(path: str) -> str:
    """
    Get the path of the latest model by replacing the placeholder with the latest version
    :param path: Path of the model with a placeholder for the version
    :return: The path to the model with the latest version
    """
    return path.replace('*', str(next_model_version(path) - 1))


def idx_to_action(action_idx: int, action_size: int) -> []:
    actions = np.zeros([action_size])
    actions[action_idx] = 1
    actions = actions.astype(int)
    return actions.tolist()


def new_episode(game: DoomGame, spawn_point_counter: Dict[int, int]) -> None:
    """
    Workaround for improper random number generation with ACS.

    In certain scenarios the agent is spawned at a random spawn point.
    However, instead of this distribution being uniform, one single id
    is heavily preferred. In order to not have the agent encounter too
    much of the same starting points, this method creates new episodes
    until one is found with a different id than the most prominent one.
    :param game: The instance of VizDoom
    :param spawn_point_counter: The dict holding the counts of the previous spawn points
    """
    while True:
        game.new_episode()
        spawn_point = game.get_game_variable(GameVariable.USER1)
        if spawn_point == 0 or spawn_point is math.isnan(spawn_point):
            return  # Spawn point undefined
        if spawn_point in spawn_point_counter:
            spawn_point_counter[spawn_point] += 1
        else:
            spawn_point_counter[spawn_point] = 0
        if spawn_point != max(spawn_point_counter, key = spawn_point_counter.get):
            return


def array_mean(values: List[float]) -> ndarray:
    return np.around(np.mean(np.array(values), dtype = "float64"), decimals = 4)


def join_stats(stats_file: TextIO, new_stats: Dict[str, float], metrics_to_rewrite: List[str]) -> Dict[str, float]:
    return {key: new_stats[key] if key in metrics_to_rewrite else value + new_stats[key]
            for key, value in json.load(stats_file).items()}


def ensure_directory(file_path: str) -> None:
    """
    Create the directory of the file if it does not exist
    :param file_path: Path to the file
    """
    directory = '/'.join(file_path.split('/')[:-1])
    if not os.path.exists(directory):
        os.mkdir(directory)
