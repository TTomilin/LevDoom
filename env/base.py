import logging
from argparse import Namespace
from collections import deque
from typing import Dict, List, Any

import skimage
from gym import spaces, Space
from gym.spaces import Discrete, Box

try:
    import vizdoom
except ImportError as e:
    raise Exception(f"Missing vizdoom package. {e}")

import gym
import numpy as np
from aenum import Enum, auto
from vizdoom import ScreenResolution, DoomGame, GameVariable

turn_off_rendering = False
try:
    from gym.envs.classic_control import rendering
except Exception as e:
    print(e)
    turn_off_rendering = True


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

        logging.info(self.game.get_available_game_variables())
        logging.info(self.game.get_available_game_variables_size())

        spaces_dict: Dict[str, spaces] = {
            'screen': Box(0, 255, self.screen_shape, dtype = np.uint8)
        }
        doom_attributes = set(map(lambda var: Scenario.DoomAttribute[var.upper()], args.game_vars))
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
        self.observation_space = spaces.Dict(spaces_dict)
        logging.info(self.game.get_available_game_variables())
        logging.info(self.game.get_available_game_variables_size())

        self.game.init()
        self.game_variable_buffer = deque(maxlen = self.variable_buffer_size)
        self.game_variable_buffer.append(self.game.get_state().game_variables)
        self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.state = None
        self.viewer = None

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
    def task_list(self) -> List[str]:
        """ List of the available tasks for the given scenario """
        raise NotImplementedError

    @property
    def n_spawn_points(self) -> int:
        """ Number of points the agents can spawn at for the given scenario """
        raise NotImplementedError

    def shape_reward(self, reward: float) -> float:
        """
        Override this method to include scenario specific reward shaping
        :param reward: The reward from the previous iteration of the game
        """
        return reward

    def additional_statistics(self, game_vars: deque) -> Dict[str, float]:
        """
        Implement this method to provide extra scenario specific statistics
        :param game_vars: Game variables of the last [variable_history_size] episodes
        :return: Dictionary of additional statistics
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

    def step(self, action):
        self.game.set_action(idx_to_action(action, self.action_space.n))
        self.game.advance_action(self.frames_per_action)

        new_state = self.game.get_state()
        self.state = new_state

        done = self.game.is_episode_finished()
        if not done:
            self.game_variable_buffer.append(new_state.game_variables)

        reward = self.game.get_last_reward()
        reward = self.shape_reward(reward)

        info = self.additional_statistics(self.game_variable_buffer)
        # self.render()
        return self._collect_observations(), reward, done, info

    def reset(self):
        if not self.game.is_running():
            self.game.init()
        self.game_variable_buffer.clear()
        self.game.new_episode()
        self.state = self.game.get_state()
        return self._collect_observations()

    def _collect_observations(self):
        observations = {}
        if self.state is not None:
            screen = np.transpose(self.state.screen_buffer, (1, 2, 0))
            screen = skimage.transform.resize(screen, self.screen_shape)
            observations['screen'] = screen
            # for variable in self.doom_variables:
            #     observations[variable.name] = np.array([self.state.game_variables[i] for i in variable.indices])
        else:
            # There is no state in the terminal step, return a "zero observations" instead
            for key, space in self.observation_space.spaces.items():
                observations[key] = np.zeros(space.shape, dtype = space.dtype)

        return observations

    def render(self, mode = "human"):
        if turn_off_rendering:
            return
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return img
        except AttributeError:
            return None

    def close(self):
        if self.viewer:
            self.viewer.close()


def idx_to_action(action_idx: int, action_size: int) -> []:
    actions = np.zeros([action_size])
    actions[action_idx] = 1
    actions = actions.astype(int)
    return actions.tolist()
