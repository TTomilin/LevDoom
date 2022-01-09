from argparse import Namespace
from collections import deque
from typing import List, Dict

import numpy as np
from aenum import Enum
from gym import Space
from gym.spaces import MultiDiscrete
from scipy.spatial import distance

from env.base import Scenario


class SeekAndKill(Scenario):

    def __init__(self, root_dir: str, task: str, args: Namespace):
        super().__init__('seek_and_kill', root_dir, task, args)
        self.max_velocity = -np.inf

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'RED', 'BLUE', 'SHADOWS', 'OBSTACLES', 'INVULNERABLE', 'MIXED_ENEMIES', 'RESIZED_ENEMIES',
                'BLUE_SHADOWS', 'OBSTACLES_RESIZED_ENEMIES', 'RED_MIXED_ENEMIES', 'INVULNERABLE_BLUE',
                'RESIZED_ENEMIES_RED', 'SHADOWS_OBSTACLES', 'BLUE_MIXED_RESIZED_ENEMIES',
                'RED_OBSTACLES_INVULNERABLE', 'RESIZED_SHADOWS_INVULNERABLE', 'COMPLETE']

    @property
    def scenario_variables(self) -> List[Scenario.DoomAttribute]:
        return [Scenario.DoomAttribute.HEALTH, Scenario.DoomAttribute.KILLS,
                Scenario.DoomAttribute.AMMO, Scenario.DoomAttribute.POSITION]

    @property
    def n_spawn_points(self) -> int:
        return 11

    @property
    def variable_buffer_size(self) -> int:
        return 5

    @property
    def statistics_fields(self) -> List[str]:
        fields = super().statistics_fields
        fields.extend(['kill_count', 'movement'])
        return fields

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variable_buffer) < 2:
            return reward  # Not enough variables in the buffer

        # Utilize a dense reward system by encouraging movement over the previous n iterations
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        dist = self.distance_traversed()
        # Increase reward linearly to the distance the agent has travelled over the past 5 frames
        reward += dist / 5000.0  # TODO make the scaling factor configurable
        # TODO make all reward scaling attributes configurable (kill count, survival, etc.)

        # TODO store each reward separately

        if current_vars[SKGameVariable.HEALTH.value] < previous_vars[SKGameVariable.HEALTH.value]:
            reward -= 0.3  # Loss of HEALTH
        if current_vars[SKGameVariable.AMMO2.value] < previous_vars[SKGameVariable.AMMO2.value]:
            reward -= 0.3  # Loss of AMMO

        return reward

    def distance_traversed(self):
        current_coords = [self.game_variable_buffer[-1][SKGameVariable.POSITION_X.value],
                          self.game_variable_buffer[-1][SKGameVariable.POSITION_Y.value]]
        past_coords = [self.game_variable_buffer[0][SKGameVariable.POSITION_X.value],  # TODO Rework retrieval
                       self.game_variable_buffer[0][SKGameVariable.POSITION_Y.value]]
        dist = distance.euclidean(current_coords, past_coords)
        return dist

    def additional_statistics(self) -> Dict[str, float]:
        return {'kill_count': self.game_variable_buffer[-1][SKGameVariable.KILL_COUNT.value],
                'movement': self.distance_traversed()}

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.KILL_COUNT

    def get_action_space(self) -> Space:
        return MultiDiscrete([
            3,  # noop, turn left, turn right
            2,  # noop, forward
            2,  # noop, shoot
            2,  # noop, sprint
        ])


class SKGameVariable(Enum):
    HEALTH = 0
    KILL_COUNT = 1
    AMMO2 = 2
    POSITION_X = 3
    POSITION_Y = 4
