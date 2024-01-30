from collections import deque
from typing import List, Dict

import numpy as np

from levdoom.envs.base import DoomEnv
from levdoom.utils.utils import distance_traversed
from levdoom.utils.wrappers import WrapperHolder, GameVariableRewardWrapper, MovementRewardWrapper


class SeekAndSlay(DoomEnv):
    """
    In this scenario, the agent is randomly spawned in one of 20 possible locations within a maze-like environment, and
    equipped with a weapon and unlimited ammunition. A fixed number of enemies are spawned at random locations at the
    beginning of an episode. Additional enemies will continually be added at random unoccupied locations after a time
    interval. The enemies are rendered immobile, forcing them to remain at their fixed locations. The goal of the agent
    is to locate and shoot the enemies. The agent can move forward, turn left and right, and shoot. The agent is granted
    a reward for each enemy killed.
    """

    def __init__(self,
                 env: str,
                 reward_kill: float = 1.0,
                 traversal_reward_scaler: float = 0.001,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.traversal_reward_scaler = traversal_reward_scaler
        self.reward_kill = reward_kill
        self.distance_buffer = []
        self.ammo_used = 0
        self.hits_taken = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        if len(game_var_buf) < 2:
            return

        distance = distance_traversed(game_var_buf, 3, 4)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
        if current_vars[2] < previous_vars[2]:
            self.ammo_used += 1

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        attack = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward:
                for a in attack:
                    actions.append(t + m + a)
        return actions

    def reward_wrappers_easy(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=1),
            WrapperHolder(MovementRewardWrapper, scaler=self.traversal_reward_scaler),
        ]

    def reward_wrappers_hard(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=1)]

    def extra_statistics(self) -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {'health': variables[0],
                'kills': variables[1],
                'ammo': self.ammo_used,
                'movement': np.mean(self.distance_buffer).round(3),
                'hits_taken': self.hits_taken}

    def clear_episode_statistics(self):
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.hits_taken = 0
        self.ammo_used = 0
