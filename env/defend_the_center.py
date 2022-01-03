from collections import deque
from typing import List, Dict

import numpy as np
from aenum import Enum

from env.base import Scenario


class DefendTheCenter(Scenario):
    """
    Scenario in which the agents stands in the center of a circular room
    with the goal of maximizing its time of survival. The agent is given
    a pistol with a limited amount of ammo to shoot approaching enemies
    who will invoke damage to agent agent as they reach melee distance.
    """

    def __init__(self, **kwargs):
        super().__init__('defend_the_center', **kwargs)

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'GORE', 'STONE_WALL', 'FAST_ENEMIES', 'MOSSY_BRICKS', 'FUZZY_ENEMIES', 'FLYING_ENEMIES',
                'RESIZED_ENEMIES', 'GORE_MOSSY_BRICKS', 'RESIZED_FUZZY_ENEMIES', 'STONE_WALL_FLYING_ENEMIES',
                'RESIZED_FLYING_ENEMIES_MOSSY_BRICKS', 'GORE_STONE_WALL_FUZZY_ENEMIES', 'FAST_RESIZED_ENEMIES_GORE',
                'COMPLETE']

    @property
    def extra_statistics(self) -> List[str]:
        return ['kills', 'ammo']

    @property
    def n_spawn_points(self) -> int:
        return 1

    @property
    def statistics_fields(self) -> []:
        fields = super().statistics_fields
        fields.extend(['kills', 'ammo'])
        return fields

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variables) < 2:
            return reward
        current_vars = self.game_variables[-1]
        previous_vars = self.game_variables[-2]
        if current_vars[DTCGameVariable.AMMO2.value] < previous_vars[DTCGameVariable.AMMO2.value]:
            reward -= 0.1  # Use of ammunition
        if current_vars[DTCGameVariable.HEALTH.value] < previous_vars[DTCGameVariable.HEALTH.value]:
            reward -= 0.1  # Loss of HEALTH
        return reward

    def additional_statistics(self, game_vars) -> Dict[str, float]:
        return {'kills': game_vars[-1][DTCGameVariable.KILL_COUNT.value],
                'ammo': game_vars[-1][DTCGameVariable.AMMO2.value]}

    def get_measurements(self, game_variables: deque, terminated: bool) -> np.ndarray:
        # TODO Determine suitable measurements for DFP
        pass

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.FRAMES_ALIVE


class DTCGameVariable(Enum):
    KILL_COUNT = 0
    AMMO2 = 1
    HEALTH = 2
