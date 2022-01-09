from argparse import Namespace
from typing import List

import gym
from aenum import Enum
from gym import Space
from gym.spaces import Discrete, MultiDiscrete

from env.base import Scenario


class HealthGathering(Scenario):

    def __init__(self, root_dir: str, task: str, args: Namespace):
        super().__init__('health_gathering', root_dir, task, args)

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'LAVA', 'SLIME', 'SUPREME', 'POISON', 'OBSTACLES', 'STIMPACKS', 'SHADED_KITS', 'WATER',
                'RESIZED_KITS', 'RESIZED_AGENT', 'SLIMY_OBSTACLES', 'SHADED_STIMPACKS', 'STIMPACKS_POISON',
                'RESIZED_KITS_LAVA', 'SUPREME_POISON', 'POISON_RESIZED_SHADED_KITS', 'OBSTACLES_SLIME_STIMPACKS',
                'LAVA_SUPREME_RESIZED_AGENT', 'COMPLETE']

    @property
    def scenario_variables(self) -> List[Scenario.DoomAttribute]:
        return [Scenario.DoomAttribute.HEALTH]

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variable_buffer) < 2:
            return reward  # Not enough variables in the buffer
        # +0.01 living reward is already configured in-game
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]
        if current_vars[HGGameVariable.HEALTH.value] > previous_vars[HGGameVariable.HEALTH.value]:
            reward += 1  # Picked up health kit
        return reward

    def get_performance_indicator(self) -> Scenario.PerformanceIndicator:
        return Scenario.PerformanceIndicator.FRAMES_ALIVE

    def get_action_space(self) -> Space:
        return MultiDiscrete([
            3,  # noop, turn left, turn right
            2,  # noop, forward
            2,  # noop, sprint
        ])


class HGGameVariable(Enum):
    HEALTH = 0
