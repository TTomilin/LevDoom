from argparse import Namespace
from typing import List, Dict, Any

import numpy as np
from gym.spaces import MultiDiscrete
from scipy import spatial
from stable_baselines3.common.logger import Logger
from vizdoom import GameVariable

from env.base import Scenario


class SeekAndKill(Scenario):

    def __init__(self, root_dir: str, task: str, args: Namespace):
        super().__init__('seek_and_kill', root_dir, task, args)
        self.max_velocity = -np.inf
        self.distance_buffer = []
        self.distance_buffer_eval = []
        self.ammo_used = 0
        self.kill_reward = args.kill_reward
        self.traversal_reward_scaler = args.traversal_reward_scaler
        self.health_loss_penalty = args.health_loss_penalty
        self.ammo_used_penalty = args.ammo_used_penalty

    @property
    def task_list(self) -> List[str]:
        return ['DEFAULT', 'RED', 'BLUE', 'SHADOWS', 'OBSTACLES', 'INVULNERABLE', 'MIXED_ENEMIES', 'RESIZED_ENEMIES',
                'BLUE_SHADOWS', 'OBSTACLES_RESIZED_ENEMIES', 'RED_MIXED_ENEMIES', 'INVULNERABLE_BLUE',
                'RESIZED_ENEMIES_RED', 'SHADOWS_OBSTACLES', 'BLUE_MIXED_RESIZED_ENEMIES',
                'RED_OBSTACLES_INVULNERABLE', 'RESIZED_SHADOWS_INVULNERABLE', 'COMPLETE']

    @property
    def scenario_variables(self) -> List[GameVariable]:
        return [GameVariable.HEALTH, GameVariable.KILLCOUNT, GameVariable.AMMO2, GameVariable.POSITION_X,
                GameVariable.POSITION_Y]

    @property
    def statistics_fields(self) -> List[str]:
        return ['kill_count', 'health', 'ammo_used', 'movement']  # The order matters

    @property
    def display_episode_length(self) -> bool:
        return False

    @property
    def n_spawn_points(self) -> int:
        return 11

    @property
    def default_variable_buffer_size(self) -> int:
        return 15

    def shape_reward(self, reward: float) -> float:
        if len(self.game_variable_buffer) < 2:
            return reward  # Not enough variables in the buffer

        # Utilize a dense reward system by encouraging movement over previous iterations
        distance = self.distance_traversed()
        self.distance_buffer.append(distance)
        reward += distance * self.traversal_reward_scaler  # Increase reward linearly

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
            self.ammo_used += 1

        return reward

    def distance_traversed(self):
        x_pos = self.field_indices['position_x']
        y_pos = self.field_indices['position_y']
        current_coords = [self.game_variable_buffer[-1][x_pos],
                          self.game_variable_buffer[-1][y_pos]]
        past_coords = [self.game_variable_buffer[0][x_pos],
                       self.game_variable_buffer[0][y_pos]]
        return spatial.distance.euclidean(current_coords, past_coords)

    def get_and_clear_episode_statistics(self) -> Dict[str, float]:
        statistics = {'kill_count': self.game_variable_buffer[-1][self.field_indices['killcount']],
                      'health': self.game_variable_buffer[-1][self.field_indices['health']],
                      'ammo_used': self.ammo_used,
                      'movement': np.mean(self.distance_buffer)}
        self.ammo_used = 0
        self.distance_buffer.clear()
        return statistics

    def log_evaluation(self, logger: Logger, statistics: Dict[str, Any]) -> None:
        logger.record("eval/health", statistics['health'])
        logger.record("eval/kill_count", statistics['kill_count'])
        logger.record("eval/ammo_used", statistics['ammo_used'])
        logger.record("eval/movement", statistics['movement'])

    def get_key_performance_indicator(self) -> Scenario.KeyPerformanceIndicator:
        return Scenario.KeyPerformanceIndicator.KILL_COUNT

    def get_multi_action_space(self) -> MultiDiscrete:
        return MultiDiscrete([
            3,  # noop, turn left, turn right
            2,  # noop, forward
            2,  # noop, shoot
            2,  # noop, sprint
        ])
