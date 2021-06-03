from collections import deque

import numpy as np
from aenum import Enum, auto, extend_enum
from scipy.spatial import distance
from typing import Dict, List
from vizdoom import ScreenResolution, DoomGame


class Scenario:
    """ Extend this abstract scenario class to define new VizDoom scenarios """

    class PerformanceIndicator(Enum):
        FRAMES_ALIVE = auto()
        KILL_COUNT = auto()

    class Task(Enum):
        DEFAULT = auto()

    def __init__(self,
                 name: str,
                 root_dir: str,
                 task: str,
                 trained_task: str,
                 window_visible: bool,
                 n_tasks: int,
                 render_hud: bool,
                 name_addition: str,
                 sound_enabled = False,
                 variable_history_size = 5,
                 screen_resolution = ScreenResolution.RES_640X480
                 ):
        self.variable_history_size = variable_history_size

        # Naming
        self.name = name
        self.root_dir = root_dir
        self.name_addition = name_addition

        # Tasks
        self.n_tasks = n_tasks
        self.task = task.lower()
        self.trained_task = trained_task

        # VizDoom
        self.game = DoomGame()
        self.game.load_config(self.config_path)
        self.game.set_doom_scenario_path(self.scenario_path)
        self.game.set_sound_enabled(sound_enabled)
        self.game.set_window_visible(window_visible)
        self.game.set_screen_resolution(screen_resolution)
        self.game.set_render_hud(render_hud)

        # Include the available tasks to the enum
        for task in self.task_list:
            extend_enum(Scenario.Task, task, auto())

    @property
    def stats_path(self) -> str:
        if self.trained_task:
            sub_folder = f'test/{self.task}/{self.trained_task}'
        elif self.n_tasks > 1:
            sub_folder = f'multi/{self.name_addition}/{self.task}' if self.name_addition else f'multi/{self.task}'
        else:
            sub_folder = f'train/{self.task}{self.name_addition}'
        return f'{self.root_dir}/statistics/{self.name}/{sub_folder}.json'

    @property
    def config_path(self) -> str:
        """ Path to the configuration file of this scenario """
        return f'{self.root_dir}/scenarios/{self.name}/{self.name}.cfg'

    @property
    def scenario_path(self) -> str:
        """ Path to the IWAD file of the designated task for this scenario """
        return f'{self.root_dir}/scenarios/{self.name}/{self.task}.wad'

    @property
    def statistics_fields(self) -> List[str]:
        """ Default metrics that will be included in the rolling statistics """
        return ['frames_alive', 'duration', 'reward']

    @property
    def task_list(self) -> List[str]:
        """ List of the available tasks for the given scenario """
        raise NotImplementedError

    @property
    def n_spawn_points(self) -> int:
        """ Number of points the agents can spawn at for the given scenario """
        raise NotImplementedError

    def shape_reward(self, reward: float, game_vars: deque) -> float:
        """
        Override this method to include scenario specific reward shaping
        :param reward: The reward from the previous iteration of the game
        :param game_vars: Game variables of the last [variable_history_size] episodes
        """
        return reward

    def additional_stats(self, game_vars: deque) -> Dict:
        """
        Implement this method to provide extra scenario specific statistics
        :param game_vars: Game variables of the last [variable_history_size] episodes
        :return: Dictionary of additional statistics
        """
        return {}

    def get_measurements(self, game_variables: deque, terminated: bool) -> np.ndarray:
        """
        Retrieve the measurement after transition for the direct future prediction algorithm
        :param game_variables: Game variables of the last [variable_history_size] iterations
        :param terminated: Indicator of whether the episode has terminated this iteration
        :return: The relevant measurements of the corresponding scenario
        """
        raise NotImplementedError

    def get_performance_indicator(self) -> PerformanceIndicator:
        """
        Every scenario has a key performance indicator (KPI) to determine the running performance
        of the agent. This is used for dynamic task prioritization, where the loss is scaled
        proportional to the task difficulty. This indicator determines the complexity of the task.
        :return: The type of the performance indicator of the implementing scenario
        """
        raise NotImplementedError


class DefendTheCenter(Scenario):
    """
    Scenario in which the agents stands in the center of a circular room
    with the goal of maximizing its time of survival. The agent is given
    a pistol with a limited amount of ammo to shoot approaching enemies
    who will invoke damage to agent agent as they reach melee distance.
    """

    def __init__(self, root_dir: str, task: str, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str):
        super().__init__('defend_the_center', root_dir, task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'GORE', 'STONE_WALL', 'FAST_ENEMIES', 'MOSSY_BRICKS', 'FUZZY_ENEMIES', 'FLYING_ENEMIES',
                'RESIZED_ENEMIES', 'GORE_MOSSY_BRICKS', 'RESIZED_FUZZY_ENEMIES', 'STONE_WALL_FLYING_ENEMIES',
                'RESIZED_FLYING_ENEMIES_MOSSY_BRICKS', 'GORE_STONE_WALL_FUZZY_ENEMIES', 'FAST_RESIZED_ENEMIES_GORE',
                'COMPLETE']

    @property
    def n_spawn_points(self) -> int:
        return 1

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

    def get_measurements(self, game_variables: deque, terminated: bool) -> np.ndarray:
        # TODO Determine suitable measurements for DFP
        pass

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.FRAMES_ALIVE


class DTCGameVariable(Enum):
    KILL_COUNT = 0
    AMMO2 = 1
    HEALTH = 2


class HealthGathering(Scenario):

    def __init__(self, root_dir: str, task: str, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str):
        super().__init__('health_gathering', root_dir, task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)
        self.health_kits = 0
        self.poison = 0

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'LAVA', 'SLIME', 'SUPREME', 'POISON', 'OBSTACLES', 'STIMPACKS', 'SHADED_KITS', 'WATER'
                'RESIZED_KITS', 'SHORT_AGENT', 'SLIMY_OBSTACLES', 'SHADED_STIMPACKS', 'STIMPACKS_POISON',
                'RESIZED_KITS_LAVA', 'SUPREME_POISON', 'POISON_RESIZED_SHADED_KITS', 'OBSTACLES_SLIME_STIMPACKS',
                'LAVA_SUPREME_SHORT_AGENT', 'COMPLETE']

    @property
    def n_spawn_points(self) -> int:
        return 1

    @property
    def statistics_fields(self) -> []:
        fields = super().statistics_fields
        fields.extend(['health_found'])
        return fields

    def get_measurements(self, game_variables: deque, terminated: bool) -> np.ndarray:
        if terminated:
            self.health_kits = 0
            self.poison = 0
        elif len(game_variables) > 1:
            current_vars = game_variables[-1]
            previous_vars = game_variables[-2]
            if previous_vars[0] - current_vars[0] > 8:  # A Poison Vial has been picked up
                self.poison += 1
            if current_vars[0] > previous_vars[0]:  # A Health Kit has been picked up
                self.health_kits += 1
        return np.array([game_variables[-1][0] / 30.0, self.health_kits / 10.0, self.poison])

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.FRAMES_ALIVE


class SeekAndKill(Scenario):

    def __init__(self, root_dir: str, task: str, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str):
        super().__init__('seek_and_kill', root_dir, task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)
        self.max_velocity = -np.inf

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'RED', 'BLUE', 'SHADOWS', 'OBSTACLES', 'INVULNERABLE', 'MIXED_ENEMIES', 'RESIZED_ENEMIES']

    @property
    def n_spawn_points(self) -> int:
        return 11

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

        # Increase reward linearly to the distance the agent has travelled over the past 5 frames
        dist = distance.euclidean(current_coords, past_coords)
        reward += dist / 500.0

        if current_vars[SKGameVariable.HEALTH.value] < previous_vars[SKGameVariable.HEALTH.value]:
            reward -= 0.3  # Loss of HEALTH
        if current_vars[SKGameVariable.AMMO2.value] < previous_vars[SKGameVariable.AMMO2.value]:
            reward -= 0.1  # Loss of AMMO

        return reward

    def additional_stats(self, game_vars) -> {}:
        return {'kill_count': game_vars[-1][SKGameVariable.KILL_COUNT.value]}

    def get_measurements(self, game_variables: deque, terminated: bool) -> np.ndarray:
        # TODO Determine suitable measurements for DFP
        pass

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.KILL_COUNT


class SKGameVariable(Enum):
    KILL_COUNT = 0
    AMMO2 = 1
    HEALTH = 2
    POSITION_X = 3
    POSITION_Y = 4


class DodgeProjectiles(Scenario):

    def __init__(self, root_dir: str, task: str, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str):
        super().__init__('dodge_projectiles', root_dir, task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition)

    @property
    def task_list(self) -> List[str]:
        return ['BARONS', 'DEFAULT', 'MANCUBUS', 'ROCK_RED', 'REVENANTS', 'CACODEMONS', 'TALL_AGENT', 'ARACHNOTRON']

    @property
    def n_spawn_points(self) -> int:
        return 1

    def shape_reward(self, reward: float, game_variables: deque) -> float:
        if len(game_variables) < 2:
            return reward
        # +0.01 living reward is already configured in-game
        current_vars = game_variables[-1]
        previous_vars = game_variables[-2]
        if current_vars[DPGameVariable.HEALTH.value] < previous_vars[DPGameVariable.HEALTH.value]:
            reward -= 1  # Loss of HEALTH
        return reward

    def get_measurements(self, game_variables: deque, terminated: bool) -> np.ndarray:
        # TODO Determine suitable measurements for DFP
        pass

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.FRAMES_ALIVE


class DPGameVariable(Enum):
    HEALTH = 0
