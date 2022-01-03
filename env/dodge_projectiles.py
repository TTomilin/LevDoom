from collections import deque
from typing import List

from aenum import Enum

from env.base import Scenario


class DodgeProjectiles(Scenario):

    def __init__(self, **kwargs):
        super().__init__('dodge_projectiles', **kwargs)

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'BARONS', 'MANCUBUS', 'FLAMES', 'CACODEMONS', 'RESIZED_AGENT', 'FLAMING_SKULLS', 'CITY',
                'REVENANTS', 'ARACHNOTRON', 'CITY_RESIZED_AGENT', 'BARONS_FLAMING_SKULLS', 'CACODEMONS_FLAMES',
                'MANCUBUS_RESIZED_AGENT', 'FLAMES_FLAMING_SKULLS_MANCUBUS', 'RESIZED_AGENT_REVENANTS',
                'CITY_ARACHNOTRON', 'COMPLETE']

    @property
    def n_spawn_points(self) -> int:
        return 1

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variables) < 2:
            return reward
        # +0.01 living reward is already configured in-game
        current_vars = self.game_variables[-1]
        previous_vars = self.game_variables[-2]
        if current_vars[DPGameVariable.HEALTH.value] < previous_vars[DPGameVariable.HEALTH.value]:
            reward -= 1  # Loss of HEALTH
        return reward

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.FRAMES_ALIVE


class DPGameVariable(Enum):
    HEALTH = 0
