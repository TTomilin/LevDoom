from argparse import Namespace
from typing import List, Dict, Any

from gym.spaces import MultiDiscrete
from stable_baselines3.common.logger import Logger
from vizdoom import GameVariable

from env.base import Scenario


class HealthGathering(Scenario):

    def __init__(self, root_dir: str, task: str, args: Namespace):
        super().__init__('health_gathering', root_dir, task, args)
        self.health_acquired_reward = args.health_acquired_reward
        self.kits_obtained = 0

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'LAVA', 'SLIME', 'SUPREME', 'POISON', 'OBSTACLES', 'STIMPACKS', 'SHADED_KITS', 'WATER',
                'RESIZED_KITS', 'RESIZED_AGENT', 'SLIMY_OBSTACLES', 'SHADED_STIMPACKS', 'STIMPACKS_POISON',
                'RESIZED_KITS_LAVA', 'SUPREME_POISON', 'POISON_RESIZED_SHADED_KITS', 'OBSTACLES_SLIME_STIMPACKS',
                'LAVA_SUPREME_RESIZED_AGENT', 'COMPLETE']

    @property
    def scenario_variables(self) -> List[Scenario.DoomAttribute]:
        return [GameVariable.HEALTH]

    @property
    def statistics_fields(self) -> List[str]:
        return ['kits_obtained']

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variable_buffer) < 2:
            return reward  # Not enough variables in the buffer

        # TODO make all reward configurable from within the benchmark
        # +0.01 living reward is already configured in-game
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]
        health_idx = self.field_indices['health']
        if current_vars[health_idx] > previous_vars[health_idx]:
            reward += self.health_acquired_reward  # Picked up health kit
            self.kits_obtained += 1
        return reward

    def get_and_clear_episode_statistics(self) -> Dict[str, float]:
        statistics = {'kits_obtained': self.kits_obtained}
        self.kits_obtained = 0
        return statistics

    def log_evaluation(self, logger: Logger, statistics: Dict[str, Any]) -> None:
        logger.record("eval/kits_obtained", statistics['kits_obtained'])

    def get_key_performance_indicator(self) -> Scenario.KeyPerformanceIndicator:
        return Scenario.KeyPerformanceIndicator.FRAMES_ALIVE

    def get_multi_action_space(self) -> MultiDiscrete:
        return MultiDiscrete([
            3,  # noop, turn left, turn right
            2,  # noop, forward
        ])

