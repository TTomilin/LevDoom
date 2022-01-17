from argparse import Namespace
from typing import List

from gym.spaces import MultiDiscrete
from vizdoom import GameVariable

from env.base import Scenario


class DodgeProjectiles(Scenario):

    def __init__(self, root_dir: str, task: str, args: Namespace):
        super().__init__('dodge_projectiles', root_dir, task, args)
        self.health_loss_penalty = args.health_loss_penalty

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'BARONS', 'MANCUBUS', 'FLAMES', 'CACODEMONS', 'RESIZED_AGENT', 'FLAMING_SKULLS', 'CITY',
                'REVENANTS', 'ARACHNOTRON', 'CITY_RESIZED_AGENT', 'BARONS_FLAMING_SKULLS', 'CACODEMONS_FLAMES',
                'MANCUBUS_RESIZED_AGENT', 'FLAMES_FLAMING_SKULLS_MANCUBUS', 'RESIZED_AGENT_REVENANTS',
                'CITY_ARACHNOTRON', 'COMPLETE']

    @property
    def scenario_variables(self) -> List[Scenario.DoomAttribute]:
        return [GameVariable.HEALTH]

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variable_buffer) < 2:
            return reward  # Not enough variables in the buffer

        # TODO make all reward configurable from within the benchmark
        # +0.01 living reward is already configured in-game
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]
        health_idx = self.field_indices['health']
        if current_vars[health_idx] < previous_vars[health_idx]:
            reward -= self.health_loss_penalty  # Loss of health
        return reward

    def get_key_performance_indicator(self) -> Scenario.KeyPerformanceIndicator:
        return Scenario.KeyPerformanceIndicator.FRAMES_ALIVE

    def get_multi_action_space(self) -> MultiDiscrete:
        return MultiDiscrete([
            3,  # noop, strafe left, strafe right
            2,  # noop, sprint
        ])
