from argparse import Namespace
from collections import deque
from typing import List, Dict

from levdoom.envs.base import DoomEnv
from levdoom.utils.wrappers import ConstantRewardWrapper, WrapperHolder, GameVariableRewardWrapper


class DodgeProjectiles(DoomEnv):
    """
    In this scenario, the agent is positioned in one end of a rectangular room, facing the opposite wall. Immobile
    enemies, equipped with projectile attacks, are lined up in front of the opposing wall, equal distance from one
    another. The objective is to survive as long as possible, ultimately until the termination of the episode. The agent
    is given no weapon nor ammunition and can only move laterally to dodge enemy projectiles. The agent is rewarded for
    each frame that it survives.
    """

    def __init__(self,
                 env: str,
                 reward_frame_survived: float = 0.01,
                 penalty_health_loss: float = -0.1,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.reward_frame_survived = reward_frame_survived
        self.penalty_health_loss = penalty_health_loss
        self.frames_survived = 0
        self.hits_taken = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1

    def reward_wrappers_hard(self) -> List[WrapperHolder]:
        return [WrapperHolder(ConstantRewardWrapper, reward=self.reward_frame_survived)]

    def reward_wrappers_easy(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_health_loss, var_index=0, decrease=True)]

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        speed = [[0.0], [1.0]]
        m_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for m in m_left_right:
            for s in speed:
                actions.append(m + s)
        return actions

    def get_statistics(self) -> Dict[str, float]:
        return {'hits_taken': self.hits_taken}

    def clear_episode_statistics(self):
        self.hits_taken = 0
