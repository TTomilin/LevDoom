import math
import os

import numpy as np
from vizdoom import GameVariable, DoomGame

from model import Algorithm


def get_input_shape(algorithm: Algorithm, img_rows, img_cols) -> tuple:
    img_channels = 3  # RGB
    trace_length = 4  # Temporal Dimension
    if algorithm == Algorithm.DRQN:
        return trace_length, img_rows, img_cols, img_channels  # 4x64x64x3
    else:
        return img_rows, img_cols, trace_length  # 64x64x4


def next_model_version(path):
    version = 0
    while os.path.exists(path.replace('*', str(version))):
        version += 1
    return version


def latest_model_path(path):
    return path.replace('*', str(next_model_version(path) - 1))


def idx_to_action(action_idx: int, action_size: int) -> []:
    actions = np.zeros([action_size])
    actions[action_idx] = 1
    actions = actions.astype(int)
    return actions.tolist()


def new_episode(game: DoomGame, spawn_point_counter: dict) -> None:
    """
    Workaround for improper random number generation with ACS.

    In certain scenarios the agent is spawned at a random spawn point.
    However, instead of this distribution being uniform, one single id
    is heavily preferred. In order to not have the agent encounter too
    much of the same starting points, this method creates new episodes
    until one is found with a different id than the most prominent one.
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
