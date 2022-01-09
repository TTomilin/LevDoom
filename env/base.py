import logging
from argparse import Namespace
from collections import deque
from typing import Dict, List, Any

import skimage
import skimage.transform
from Cython.Utils import OrderedSet
from gym import spaces, Space
from gym.spaces import Discrete, Box, MultiDiscrete

try:
    import vizdoom
except ImportError as e:
    raise Exception(f"Missing vizdoom package. {e}")

import gym
import numpy as np
from aenum import Enum, auto
from vizdoom import ScreenResolution, DoomGame, GameVariable

try:
    from gym.envs.classic_control import rendering
except Exception as e:
    logging.error(f'Rendering import failed: {e}')


class DoomVariable:
    def __init__(self, name: str, game_variables: List[GameVariable], space: Space):
        self.name = name
        self.game_variables = game_variables
        self.space = space
        self.buffer = []
        self.indices = []

    def get_buffer(self):
        return self.buffer

    def add_variable(self, var: Any):
        return self.buffer.append(var)

    def add_index(self, index: int):
        return self.indices.append(index)


class Scenario(gym.Env):
    """ Extend this abstract scenario class to define new ViZDoom scenarios """

    class DoomAttribute(Enum):
        POSITION = ([GameVariable.POSITION_X, GameVariable.POSITION_Y, GameVariable.POSITION_Z],
                    Box(-np.Inf, np.Inf, (3, 1))),
        VELOCITY = ([GameVariable.VELOCITY_X, GameVariable.VELOCITY_Y, GameVariable.VELOCITY_Z],
                    Box(-np.Inf, np.Inf, (3, 1))),
        HEALTH = ([GameVariable.HEALTH], Box(0, np.Inf, (1, 1))),
        KILLS = ([GameVariable.KILLCOUNT], Discrete(1)),
        AMMO = ([GameVariable.AMMO2], Discrete(1)),

    class PerformanceIndicator(Enum):
        FRAMES_ALIVE = auto()
        KILL_COUNT = auto()

    class Task(Enum):
        DEFAULT = auto()

    def __init__(self, name: str, root_dir: str, task: str, args: Namespace):
        self.variable_history_size = args.variable_history_size
        self.frames_per_action = args.frames_per_action

        # Naming
        self.name = name
        self.root_dir = root_dir
        self.name_addition = args.model_name_addition

        # Tasks
        self.n_tasks = len(args.tasks)
        self.task = task.lower()
        self.trained_task = None

        # VizDoom
        game = DoomGame()
        game.load_config(self.config_path)
        game.set_doom_scenario_path(self.scenario_path)
        game.set_sound_enabled(args.sound)
        game.set_window_visible(args.visualize)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_render_hud(args.render_hud)

        self.screen_shape = (args.frame_height, args.frame_width, game.get_screen_channels())
        self.game = game

        # spaces_dict: Dict[str, spaces] = {
        #     'screen': Box(0, 255, self.screen_shape, dtype = np.uint8)
        # }
        doom_attributes = OrderedSet(map(lambda var: Scenario.DoomAttribute[var.upper()], args.game_vars))
        doom_attributes.update(self.scenario_variables)
        self.doom_variables = list(map(lambda attr: DoomVariable(attr.name.lower(), attr.value[0][0], attr.value[0][1]),
                                       list(doom_attributes)))
        # self.doom_attributes = deepcopy(self.doom_attributes)
        index = 0
        self.game.clear_available_game_variables()
        for variable in self.doom_variables:
            for label in variable.game_variables:
                self.game.add_available_game_variable(label)
                variable.indices.append(index)
                index += 1
            # spaces_dict[variable.name] = variable.space
        # self.observation_space = spaces.Dict(spaces_dict)
        self.observation_space = Box(0, 255, self.screen_shape, dtype = np.uint8)
        self.game.init()
        self.game_variable_buffer = deque(maxlen = self.variable_buffer_size)
        self.game_variable_buffer.append(self.game.get_state().game_variables)
        # self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.action_space = self.get_action_space()
        self.state = None
        self.viewer = None

    @property
    def task_list(self) -> List[str]:
        """ List of the available tasks for the given scenario """
        raise NotImplementedError

    @property
    def config_path(self) -> str:
        """ Path to the configuration file of this scenario """
        return f'{self.root_dir}/scenarios/{self.name}/{self.name}.cfg'

    @property
    def scenario_path(self) -> str:
        """ Path to the IWAD file of the designated task for this scenario """
        return f'{self.root_dir}/scenarios/{self.name}/{self.task}.wad'

    @property
    def extra_statistics(self) -> List[str]:
        """ Extra metrics that will be included in the rolling statistics """
        return []

    @property
    def scenario_variables(self) -> List[DoomAttribute]:
        """ Default scenario specific game variables that will be returned from the ViZDoom environment """
        return []

    @property
    def variable_buffer_size(self) -> int:
        """ Game variables of the last n iterations; used for reward shaping; default = 2 --> current & previous """
        return 2

    @property
    def statistics_fields(self) -> List[str]:
        return []

    @property
    def n_spawn_points(self) -> int:
        """ Number of locations where the agent may spawn """
        return 1

    def shape_reward(self, reward: float) -> float:
        """
        Override this method to include scenario specific reward shaping
        :param reward: The reward from the previous iteration of the game
        """
        return reward

    def additional_statistics(self) -> Dict[str, float]:
        """
        Implement this method to provide extra scenario specific statistics
        :return: Dictionary of additional statistics. Empty by default
        """
        return {}

    def get_performance_indicator(self) -> PerformanceIndicator:
        """
        Every scenario has a key performance indicator (KPI) to determine the running performance
        of the agent. This is used for dynamic task prioritization, where the loss is scaled
        proportional to the task difficulty. This indicator determines the complexity of the task.
        :return: The type of the performance indicator of the implementing scenario
        """
        raise NotImplementedError

    def get_action_space(self) -> Space:
        """
        :return: The action space of the scenario
        """
        raise NotImplementedError

    def step(self, action):
        actions_flattened = self._convert_actions(action)
        self.game.set_action(actions_flattened)
        self.game.advance_action(self.frames_per_action)

        new_state = self.game.get_state()
        self.state = new_state

        done = self.game.is_episode_finished()
        if not done:
            self.game_variable_buffer.append(new_state.game_variables)

        reward = self.game.get_last_reward()
        reward = self.shape_reward(reward)

        info = self.additional_statistics()
        return self._collect_observations(), reward, done, info

    def reset(self):
        if not self.game.is_running():
            self.game.init()
        self.game_variable_buffer.clear()
        self.game.new_episode()
        self.state = self.game.get_state()
        return self._collect_observations()

    def render(self, mode = "human"):
        img = self.game.get_state().screen_buffer
        img = np.transpose(img, [1, 2, 0])
        if mode == 'rgb_array':
            return img  # return RGB frame suitable for video
        elif mode == 'human':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return img
        else:
            return super().render(mode)  # Not Implemented

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _convert_actions(self, actions):
        """Convert actions from gym action space to the action space expected by Doom game."""

        # if self.composite_action_space:
        #     # composite action space with multiple subspaces
        #     spaces = self.action_space.spaces
        # else:
        #     # simple action space, e.g. Discrete. We still treat it like composite of length 1
        #     spaces = (self.action_space, )
        #     actions = (actions, )
        space_type = type(self.action_space)
        if space_type == MultiDiscrete:
            spaces = self.action_space.nvec
            actions_flattened = []
            for i, action in enumerate(actions):
                num_non_idle_actions = spaces[i] - 1
                action_one_hot = np.zeros(num_non_idle_actions, dtype=np.uint8)
                if action > 0:
                    action_one_hot[action - 1] = 1  # 0th action in each subspace is a no-op
                actions_flattened.extend(action_one_hot)
            return actions_flattened
        elif space_type == Discrete:
            return self._idx_to_action(actions)
        else:
            raise NotImplementedError(f'Action space {space_type} is not supported')

    def _collect_observations(self):
        if self.state is None:
            return np.zeros(self.observation_space.shape, dtype = self.observation_space.dtype)
        else:
            screen = np.transpose(self.state.screen_buffer, (1, 2, 0))
            return skimage.transform.resize(screen, self.screen_shape)

    # def _collect_observations(self):
    #     observations = {}
    #     if self.state is not None:
    #         screen = np.transpose(self.state.screen_buffer, (1, 2, 0))
    #         screen = skimage.transform.resize(screen, self.screen_shape)
    #         observations['screen'] = screen
    #         # for variable in self.doom_variables:
    #         #     observations[variable.name] = np.array([self.state.game_variables[i] for i in variable.indices])
    #     else:
    #         # There is no state in the terminal step, return a "zero observations" instead
    #         for key, space in self.observation_space.spaces.items():
    #             observations[key] = np.zeros(space.shape, dtype = space.dtype)
    #
    #     return observations

    def _idx_to_action(self, action_idx: int) -> List[int]:
        actions = np.zeros([self.action_space.n])
        actions[action_idx] = 1
        actions = actions.astype(int)
        return actions.tolist()
