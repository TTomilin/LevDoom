import gymnasium
import numpy as np
from gymnasium.core import ObsType, WrapperObsType, RewardWrapper
from gymnasium.spaces import Box


class RescaleObservation(gymnasium.ObservationWrapper):
    """Rescale the observation space to [-1, 1]."""

    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.observation_space = Box(low=-1, high=1, shape=self.observation_space.shape)

    def observation(self, observation: ObsType) -> WrapperObsType:
        return observation / 255. * 2 - 1


class RGBStack(gymnasium.ObservationWrapper):
    """Combine the stacked frames with RGB colours. [n_stack, h, w, 3] -> [n_stack * 3, h, w]"""

    def __init__(self, env):
        super(RGBStack, self).__init__(env)
        n_stack, height, width, channels = self.observation_space.shape
        new_shape = (n_stack * channels, height, width)
        self.observation_space = Box(
            low=np.min(self.observation_space.low),
            high=np.max(self.observation_space.high),
            shape=new_shape
        )

    def observation(self, observation: ObsType) -> WrapperObsType:
        n_stack, height, width, channels = observation.shape
        return np.transpose(observation, (0, 3, 1, 2)).reshape(n_stack * channels, height, width)


class WrapperHolder:
    """
    A wrapper holder stores a reward wrapper with its respective keyword arguments.
    """

    def __init__(self, wrapper_class, **kwargs):
        self.wrapper_class = wrapper_class
        self.kwargs = kwargs


class ConstantRewardWrapper(RewardWrapper):
    """
    Reward the agent with a constant reward at each time step.
    """

    def __init__(self, env, reward: float):
        super(ConstantRewardWrapper, self).__init__(env)
        self.rew = reward

    def reward(self, reward: float) -> float:
        return reward + self.rew


class GameVariableRewardWrapper(RewardWrapper):
    """
    Reward the agent for a change in a game variable. The agent is considered to have changed a game variable if its
    value differs from the previous frame value.
    """

    def __init__(self, env, reward: float, var_index: int = 0, decrease: bool = False):
        super(GameVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_index = var_index
        self.decrease = decrease

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward
        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        var_cur = vars_cur[self.var_index]
        var_prev = vars_prev[self.var_index]

        if not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            reward += self.rew
        return reward


class MovementRewardWrapper(RewardWrapper):
    """
    Reward the agent for moving. Movement is measured as the distance between the agent's current location and its
    location in the previous frame.
    """

    def __init__(self, env, scaler: float):
        super(MovementRewardWrapper, self).__init__(env)
        self.scaler = scaler

    def reward(self, reward):
        if len(self.distance_buffer) < 2:
            return reward
        distance = self.distance_buffer[-1]
        reward += distance * self.scaler  # Increase the reward for movement linearly
        return reward
