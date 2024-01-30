from argparse import Namespace
from collections import deque
from typing import List, Dict

import numpy as np

from levdoom.envs.base import DoomEnv
from levdoom.utils.utils import distance_traversed
from levdoom.utils.wrappers import WrapperHolder, ConstantRewardWrapper, GameVariableRewardWrapper


class HealthGathering(DoomEnv):
    """
    In this scenario, the agent is trapped in a room with a surface, which slowly but constantly decreases the agent’s
    health. Health granting items continually spawn in random locations at specified time intervals. The default health
    item heals grants 25 hit points. Some environments contain poison vials, which inflict damage to the agent instead
    of providing health. The objective is to survive. The agent should identify the health granting items and navigate
    around the map to collect them quickly enough to avoid running out of health. The agent can turn left and right,
    and move forward. A small reward is granted for every frame the agent manages to survive.
    """
    def __init__(self,
                 env: str,
                 reward_health_kit: float = 1.0,
                 reward_frame_survived: float = 0.01,
                 penalty_health_loss: float = -0.01,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.reward_frame_survived = reward_frame_survived
        self.reward_health_kit = reward_health_kit
        self.penalty_health_loss = penalty_health_loss
        self.distance_buffer = []
        self.frames_survived = 0
        self.kits_obtained = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return

        distance = distance_traversed(game_var_buf, 1, 2)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] > previous_vars[0]:
            self.kits_obtained += 1

    def reward_wrappers_hard(self) -> List[WrapperHolder]:
        return [WrapperHolder(ConstantRewardWrapper, reward=self.reward_frame_survived)]

    def reward_wrappers_easy(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, reward=self.penalty_health_loss),
            WrapperHolder(GameVariableRewardWrapper, reward=self.reward_health_kit, var_index=0),
        ]

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for m in m_forward:
            for t in t_left_right:
                actions.append(t + m)
        return actions

    def get_statistics(self) -> Dict[str, float]:
        return {'kits_obtained': self.kits_obtained,
                'movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self):
        self.kits_obtained = 0
