from argparse import Namespace
from typing import List

from aenum import Enum
from gym.spaces import MultiDiscrete

from env.base import Scenario


class DodgeProjectiles(Scenario):

    def __init__(self, root_dir: str, task: str, args: Namespace, multi_action=True):
        super().__init__('dodge_projectiles', root_dir, task, args, multi_action)

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'BARONS', 'MANCUBUS', 'FLAMES', 'CACODEMONS', 'RESIZED_AGENT', 'FLAMING_SKULLS', 'CITY',
                'REVENANTS', 'ARACHNOTRON', 'CITY_RESIZED_AGENT', 'BARONS_FLAMING_SKULLS', 'CACODEMONS_FLAMES',
                'MANCUBUS_RESIZED_AGENT', 'FLAMES_FLAMING_SKULLS_MANCUBUS', 'RESIZED_AGENT_REVENANTS',
                'CITY_ARACHNOTRON', 'COMPLETE']

    @property
    def scenario_variables(self) -> List[Scenario.DoomAttribute]:
        return [Scenario.DoomAttribute.HEALTH]

    @property
    def statistics_fields(self) -> List[str]:
        return ['health']

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variable_buffer) < 2:
            return reward  # Not enough variables in the buffer

        # +0.01 living reward is already configured in-game
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]
        if current_vars[DPGameVariable.HEALTH.value] < previous_vars[DPGameVariable.HEALTH.value]:
            reward -= 1  # Loss of HEALTH
        return reward

    def get_key_performance_indicator(self) -> Scenario.KeyPerformanceIndicator:
        return Scenario.KeyPerformanceIndicator.FRAMES_ALIVE

    def get_multi_action_space(self) -> MultiDiscrete:
        return MultiDiscrete([
            3,  # noop, strafe left, strafe right
            2,  # noop, sprint
        ])


# TODO Deprecate
class DPGameVariable(Enum):
    HEALTH = 0
