from argparse import Namespace
from typing import List, Dict, Any

from gym.spaces import MultiDiscrete
from stable_baselines3.common.logger import Logger
from vizdoom import GameVariable

from env.base import Scenario


class DefendTheCenter(Scenario):
    """
    Scenario in which the agent stands in the center of a circular room
    with the goal of maximizing its time of survival. The agent is given
    a pistol with a limited amount of ammo to shoot approaching enemies
    who will invoke damage to the agent as they reach melee distance.
    """

    def __init__(self, root_dir: str, task: str, args: Namespace):
        super().__init__('defend_the_center', root_dir, task, args)
        self.kill_reward = args.kill_reward
        self.health_loss_penalty = args.health_loss_penalty
        self.ammo_used_penalty = args.ammo_used_penalty

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'GORE', 'STONE_WALL', 'FAST_ENEMIES', 'MOSSY_BRICKS', 'FUZZY_ENEMIES', 'FLYING_ENEMIES',
                'RESIZED_ENEMIES', 'GORE_MOSSY_BRICKS', 'RESIZED_FUZZY_ENEMIES', 'STONE_WALL_FLYING_ENEMIES',
                'RESIZED_FLYING_ENEMIES_MOSSY_BRICKS', 'GORE_STONE_WALL_FUZZY_ENEMIES', 'FAST_RESIZED_ENEMIES_GORE',
                'COMPLETE']

    @property
    def scenario_variables(self) -> List[Scenario.DoomAttribute]:
        return [GameVariable.KILLCOUNT, GameVariable.AMMO2, GameVariable.HEALTH]

    @property
    def statistics_fields(self) -> List[str]:
        return ['kill_count', 'ammo_left']  # The order matters

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variable_buffer) < 2:
            return reward  # Not enough variables in the buffer

        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        kc_idx = self.field_indices['killcount']
        health_idx = self.field_indices['health']
        ammo_idx = self.field_indices['ammo2']

        if current_vars[kc_idx] > previous_vars[kc_idx]:
            reward += self.kill_reward  # Elimination of enemy
        if current_vars[health_idx] < previous_vars[health_idx]:
            reward -= self.health_loss_penalty  # Loss of health
        if current_vars[ammo_idx] < previous_vars[ammo_idx]:
            reward -= self.ammo_used_penalty  # Use of ammunition

        return reward

    def get_and_clear_episode_statistics(self) -> Dict[str, float]:
        return {'kill_count': self.game_variable_buffer[-1][self.field_indices['killcount']],
                'ammo_left': self.game_variable_buffer[-1][self.field_indices['ammo2']]}

    def log_evaluation(self, logger: Logger, statistics: Dict[str, Any]) -> None:
        logger.record("eval/kill_count", statistics['kill_count'])
        logger.record("eval/ammo_left", statistics['ammo_left'])

    def get_key_performance_indicator(self) -> Scenario.KeyPerformanceIndicator:
        return Scenario.KeyPerformanceIndicator.FRAMES_ALIVE

    def get_multi_action_space(self) -> MultiDiscrete:
        return MultiDiscrete([
            3,  # noop, turn left, turn right
            2,  # noop, shoot
        ])
