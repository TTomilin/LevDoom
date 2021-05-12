from collections import deque
from ctypes import Array
from enum import Enum, auto

import numpy as np
from scipy.spatial import distance
from typing import Dict, List
from vizdoom import *

from model import Algorithm


class Scenario:
    """ Generic Scenario.
    Extend this abstract scenario class to define new VizDoom scenarios
    """

    class SubTask(Enum):
        DEFAULT = auto()

    def __init__(self,
                 name: str,
                 base_dir: str,
                 algorithm: Algorithm,
                 sub_task: SubTask,
                 trained_task: str,
                 window_visible: bool,
                 n_tasks: int,
                 render_hud: bool,
                 name_addition: str,
                 sound_enabled = False,
                 len_vars_history = 5,
                 screen_resolution = ScreenResolution.RES_640X480
                 ):
        self.len_vars_history = len_vars_history

        # Naming
        self.name = name
        self.base_dir = base_dir
        self.name_addition = name_addition
        self.alg_name = algorithm.name.lower()

        # Tasks
        self.n_tasks = n_tasks
        self.task = sub_task.name.lower()
        self.trained_task = trained_task

        # VizDoom
        self.game = DoomGame()
        self.game.load_config(self.config_path)
        self.game.set_doom_scenario_path(self.scenario_path)
        self.game.set_sound_enabled(sound_enabled)
        self.game.set_window_visible(window_visible)
        self.game.set_screen_resolution(screen_resolution)
        self.game.set_render_hud(render_hud)

    @property
    def full_name(self) -> str:
        return f'{self.name}_{self.task}'

    @property
    def stats_path(self) -> str:
        if self.trained_task:
            sub_folder = f'test/{self.task}/{self.trained_task}'
        elif self.n_tasks > 1:
            sub_folder = f'multi/{self.name_addition}/{self.task}' if self.name_addition else f'multi/{self.task}'
        else:
            sub_folder = f'train/{self.task}{self.name_addition}'
        return f'{self.base_dir}statistics/{self.name}/{sub_folder}.json'

    @property
    def config_path(self) -> str:
        return f'{self.base_dir}scenarios/{self.name}/{self.name}.cfg'

    @property
    def scenario_path(self) -> str:
        return f'{self.base_dir}scenarios/{self.name}/{self.full_name}.wad'

    @property
    def statistics_fields(self) -> List[str]:
        return ['frames_alive', 'duration', 'reward']

    def shape_reward(self, reward: float, game_vars: deque) -> float:
        return reward

    def additional_stats(self, game_vars: deque) -> Dict:
        """
        Implement this method to provide extra scenario specific statistics
        :param game_vars: Game variables of the last [len_vars_history] episodes
        :return: Dictionary of added statistics
        """
        return {}

    def get_measurements(self, game_vars: deque, **kwargs) -> np.ndarray:
        """
        Retrieve the measurement after transition for the direct future prediction algorithm
        :param game_vars: Game variables of the last [len_vars_history] episodes
        :param kwargs: Additional keyword arguments to calculate the measurements
        :return: The relevant measurements of the corresponding scenario
        """
        raise NotImplementedError


class DefendTheCenter(Scenario):
    """
    Scenario in which the agents stands in the center of a circular room
    with the goal of maximizing its time of survival. The agent is given
    a pistol with a limited amount of ammo to shoot approaching enemies
    who will invoke damage to agent agent as they reach melee distance.
    """

    class SubTask(Enum):
        GORE = auto()
        MULTI = auto()
        DEFAULT = auto()
        STONE_WALL = auto()
        FAST_ENEMIES = auto()
        MOSSY_BRICKS = auto()
        FUZZY_ENEMIES = auto()
        FLYING_ENEMIES = auto()
        RESIZED_ENEMIES = auto()

    def __init__(self, base_dir: str, algorithm: Algorithm, sub_task: SubTask, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str) -> Scenario:
        super().__init__('defend_the_center', base_dir, algorithm, sub_task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)

    @property
    def statistics_fields(self) -> []:
        fields = super().statistics_fields
        fields.extend(['kill_count', 'ammo_left'])
        return fields

    def shape_reward(self, reward: float, game_variables: deque) -> float:
        if len(game_variables) < 2:
            return reward
        current_vars = game_variables[-1]
        previous_vars = game_variables[-2]
        if current_vars[DTCGameVariable.AMMO2.value] < previous_vars[DTCGameVariable.AMMO2.value]:
            reward -= 0.1  # Use of ammunition
        if current_vars[DTCGameVariable.HEALTH.value] < previous_vars[DTCGameVariable.HEALTH.value]:
            reward -= 0.3  # Loss of HEALTH
        return reward

    def additional_stats(self, game_vars) -> {}:
        return {'kill_count': game_vars[-1][SKGameVariable.KILL_COUNT.value],
                'ammo_left': game_vars[-1][DTCGameVariable.AMMO2.value]}

    def get_measurements(self, game_vars: Array, **kwargs) -> np.ndarray:
        # TODO Determine suitable measurements for DFP
        pass


class DTCGameVariable(Enum):
    KILL_COUNT = 0
    AMMO2 = 1
    HEALTH = 2


class HealthGathering(Scenario):
    class SubTask(Enum):
        LAVA = auto()
        MULTI = auto()
        DEFAULT = auto()
        SUPREME = auto()
        OBSTACLES = auto()
        STIMPACKS = auto()
        SHADED_KITS = auto()
        RESIZED_KITS = auto()
        STIMPACKS_POISON = auto()

    def __init__(self, base_dir: str, algorithm: Algorithm, sub_task: SubTask, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str) -> Scenario:
        super().__init__('health_gathering', base_dir, algorithm, sub_task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)

    @property
    def statistics_fields(self) -> []:
        fields = super().statistics_fields
        fields.extend(['health_found'])
        return fields

    def get_measurements(self, game_variables: Array, **kwargs) -> np.ndarray:
        return np.array([game_variables[-1][0] / 30.0, kwargs['health_kit'] / 10.0, kwargs['poison']])


class SeekAndKill(Scenario):
    class SubTask(Enum):
        RED = auto()
        BLUE = auto()
        MULTI = auto()
        SHADOWS = auto()
        DEFAULT = auto()
        OBSTACLES = auto()
        INVULNERABLE = auto()
        MIXED_ENEMIES = auto()
        RESIZED_ENEMIES = auto()

    def __init__(self, base_dir: str, algorithm: Algorithm, sub_task: SubTask, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str) -> Scenario:
        super().__init__('seek_and_kill', base_dir, algorithm, sub_task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)
        self.max_velocity = -np.inf

    @property
    def statistics_fields(self) -> []:
        fields = super().statistics_fields
        fields.extend(['kill_count', 'ammo_left'])
        return fields

    def shape_reward(self, reward: float, game_variables: deque) -> float:
        if len(game_variables) < 2:
            return reward
        current_vars = game_variables[-1]
        previous_vars = game_variables[-2]

        current_coords = [current_vars[SKGameVariable.POSITION_X.value], current_vars[SKGameVariable.POSITION_Y.value]]
        past_coords = [game_variables[0][SKGameVariable.POSITION_X.value],
                       game_variables[0][SKGameVariable.POSITION_Y.value]]
        dist = distance.euclidean(current_coords, past_coords)
        distance_threshold = 50

        if current_vars[SKGameVariable.HEALTH.value] < previous_vars[SKGameVariable.HEALTH.value]:
            reward -= 0.3  # Loss of HEALTH
        if current_vars[SKGameVariable.AMMO2.value] < previous_vars[SKGameVariable.AMMO2.value]:
            reward -= 0.1  # Loss of AMMO
        if dist > distance_threshold and len(game_variables) == self.len_vars_history:
            reward += 0.2  # Encourage movement

        return reward

    def additional_stats(self, game_vars) -> {}:
        return {'kill_count': game_vars[-1][SKGameVariable.KILL_COUNT.value]}

    def get_measurements(self, game_vars: Array, **kwargs) -> np.ndarray:
        # TODO Determine suitable measurements for DFP
        pass


class SKGameVariable(Enum):
    KILL_COUNT = 0
    AMMO2 = 1
    HEALTH = 2
    POSITION_X = 3
    POSITION_Y = 4


class DodgeProjectiles(Scenario):
    class SubTask(Enum):
        MULTI = auto()
        BARONS = auto()
        DEFAULT = auto()
        MANCUBUS = auto()
        ROCK_RED = auto()
        REVENANTS = auto()
        CACODEMONS = auto()
        TALL_AGENT = auto()
        ARACHNOTRON = auto()

    def __init__(self, base_dir: str, algorithm: Algorithm, sub_task: SubTask, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str) -> Scenario:
        super().__init__('dodge_projectiles', base_dir, algorithm, sub_task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)

    def shape_reward(self, reward: float, game_variables: deque) -> float:
        if len(game_variables) < 2:
            return reward
        # +0.01 living reward already configured
        current_vars = game_variables[-1]
        previous_vars = game_variables[-2]
        if current_vars[DPGameVariable.HEALTH.value] < previous_vars[DPGameVariable.HEALTH.value]:
            reward -= 1  # Loss of HEALTH
        return reward

    def get_measurements(self, game_vars: Array, **kwargs) -> np.ndarray:
        # TODO Determine suitable measurements for DFP
        pass


class DPGameVariable(Enum):
    HEALTH = 0
