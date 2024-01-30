from collections import deque
from typing import List, Dict

from levdoom.envs.base import DoomEnv
from levdoom.utils.wrappers import WrapperHolder, GameVariableRewardWrapper


class DefendTheCenter(DoomEnv):
    """
    In this scenario, the agent is spawned in the center of a circular room. Enemies are spawned at fixed positions
    alongside the wall of the area. The enemies do not possess a projectile attack and therefore have to make their way
    within melee range to inflict damage. The agent is rendered immobile, but equipped with a weapon and limited
    ammunition to fend off the encroaching enemies. Once the enemies are killed, they respawn at their original location
    after a fixed time delay. The objective of the agent is to survive as long as possible. The agent is rewarded for
    each enemy killed.
    """

    def __init__(self,
                 env: str,
                 reward_kill: float = 1.0,
                 penalty_health_loss: float = -0.01,
                 penalty_ammo_used: float = -0.01,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.reward_kill = reward_kill
        self.penalty_health_loss = penalty_health_loss
        self.penalty_ammo_used = penalty_ammo_used
        self.frames_survived, self.hits_taken = 0, 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[1] < previous_vars[1]:
            self.hits_taken += 1

    def reward_wrappers_hard(self) -> List[WrapperHolder]:
        return [WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=0)]

    def reward_wrappers_easy(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, reward=self.reward_kill, var_index=0),
            WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_health_loss, var_index=1, decrease=True),
            WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_ammo_used, var_index=2, decrease=True),
        ]

    def get_available_actions(self) -> List[List[float]]:
        actions = []
        attack = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for t in t_left_right:
            for a in attack:
                actions.append(t + a)
        return actions

    def extra_statistics(self) -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {'kills': variables[0],
                'health': variables[1],
                'ammo_left': variables[2],
                'hits_taken': self.hits_taken}

    def clear_episode_statistics(self):
        self.hits_taken = 0
