from collections import deque
from typing import List

import numpy as np
from aenum import Enum
from scipy.spatial import distance

from env.base import Scenario


class SeekAndKill(Scenario):

    def __init__(self, root_dir: str, task: str, trained_task: str,
                 window_visible: bool, n_tasks: int, render_hud: bool, name_addition: str, sound_enabled=False):
        super().__init__('seek_and_kill', root_dir, task, trained_task, window_visible, n_tasks,
                         render_hud, name_addition, sound_enabled)
        self.max_velocity = -np.inf

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'RED', 'BLUE', 'SHADOWS', 'OBSTACLES', 'INVULNERABLE', 'MIXED_ENEMIES', 'RESIZED_ENEMIES',
                'BLUE_SHADOWS', 'OBSTACLES_RESIZED_ENEMIES', 'RED_MIXED_ENEMIES', 'INVULNERABLE_BLUE',
                'RESIZED_ENEMIES_RED', 'SHADOWS_OBSTACLES', 'BLUE_MIXED_RESIZED_ENEMIES',
                'RED_OBSTACLES_INVULNERABLE', 'RESIZED_SHADOWS_INVULNERABLE', 'COMPLETE']

    @property
    def n_spawn_points(self) -> int:
        return 11

    @property
    def variable_buffer_size(self) -> int:
        return 5

    @property
    def statistics_fields(self) -> []:
        fields = super().statistics_fields
        fields.extend(['kill_count', 'ammo_left'])
        return fields

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variables) < 2:
            return reward

        # Utilize a dense reward system by encouraging movement over the previous n iterations
        current_vars = self.game_variables[-1]
        previous_vars = self.game_variables[-2]

        current_coords = [current_vars[SKGameVariable.POSITION_X.value], current_vars[SKGameVariable.POSITION_Y.value]]
        past_coords = [self.game_variables[0][SKGameVariable.POSITION_X.value],
                       self.game_variables[0][SKGameVariable.POSITION_Y.value]]

        # Increase reward linearly to the distance the agent has travelled over the past 5 frames
        dist = distance.euclidean(current_coords, past_coords)
        reward += dist / 500.0

        if current_vars[SKGameVariable.HEALTH.value] < previous_vars[SKGameVariable.HEALTH.value]:
            reward -= 0.3  # Loss of HEALTH
        if current_vars[SKGameVariable.AMMO2.value] < previous_vars[SKGameVariable.AMMO2.value]:
            reward -= 0.1  # Loss of AMMO

        return reward

    def additional_statistics(self, game_vars) -> {}:
        return {'kill_count': game_vars[-1][SKGameVariable.KILL_COUNT.value]}

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.KILL_COUNT


class SKGameVariable(Enum):
    KILL_COUNT = 0
    AMMO2 = 1
    HEALTH = 2
    POSITION_X = 3
    POSITION_Y = 4
