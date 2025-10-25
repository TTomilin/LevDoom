from collections import deque
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import cv2
import gymnasium
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution, GameVariable

from levdoom.utils.utils import get_screen_resolution


class DoomEnv(gymnasium.Env):

    def __init__(self,
                 env: str = 'default',
                 frame_skip: int = 4,
                 seed: int = 0,
                 render: bool = True,
                 resolution: str = None,
                 max_steps: int = None,
                 variable_queue_length: int = 5):
        super().__init__()
        self.env = env
        self.frame_skip = frame_skip
        self.scenario = self.__module__.split('.')[-2]

        # Determine the directory of the doom scenario
        scenario_dir = f'{Path(__file__).parent.resolve()}/{self.scenario}'

        # Create the Doom game instance
        self.game = vzd.DoomGame()
        self.game.load_config(f"{scenario_dir}/conf.cfg")
        self.game.set_doom_scenario_path(f"{scenario_dir}/maps/{env}.wad")
        self.game.set_seed(seed)
        self.render_enabled = render
        if max_steps:
            self.game.set_episode_timeout(max_steps)
        if render:
            # Use a higher resolution for rendering gameplay
            self.game.set_screen_resolution(ScreenResolution.RES_1600X1200)
            self.frame_skip = 1
        elif resolution:  # Use a particular predefined resolution
            self.game.set_screen_resolution(get_screen_resolution(resolution))
        self.game.init()

        # Define the observation space
        self.game_res = (self.game.get_screen_height(), self.game.get_screen_width(), 3)
        self._observation_space = gymnasium.spaces.Box(low=0, high=255, shape=self.game_res, dtype=np.uint8)

        # Define the action space
        self.available_actions = self.get_available_actions()
        self._action_space = gymnasium.spaces.Discrete(len(self.available_actions))

        # Initialize and fill the game variable queue
        self.game_variable_buffer = deque(maxlen=variable_queue_length)
        self.game_variable_buffer.append(self.game.get_state().game_variables)

        # self.extra_statistics = ['kills', 'health', 'ammo', 'movement', 'kits_obtained', 'hits_taken']

    @property
    def name(self) -> str:
        return f'{self.scenario}-{self.env}'

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return self._action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return self._observation_space

    def extra_statistics(self) -> Dict[str, float]:
        """
        Retrieves additional statistics specific to the scenario. Mostly game variables.

        Args:
            mode (str): A specifier to distinguish which environment the statistic is for (train/test).

        Returns:
            statistics (Dict[str, float]): A dictionary containing additional statistical data.
        """
        return {}

    def store_statistics(self, game_vars: deque) -> None:
        """
        Stores statistics based on the game variables.

        Args:
            game_vars (deque): A deque containing game variables for statistics.
        """
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state and returns the initial observation.

        Args:
            seed (Optional[int]): Seed for random number generator.
            options (Optional[dict]): Additional options for environment reset.

        Returns:
            observation (np.ndarray): Initial state observation of the environment.
            info (Dict[str, Any]): Additional information about the initial state.
        """
        try:
            self.game.new_episode()
        except vzd.ViZDoomIsNotRunningException:
            print('ViZDoom is not running. Restarting...')
            self.game.init()
            self.game.new_episode()
        self.clear_episode_statistics()
        state = self.game.get_state().screen_buffer
        state = np.transpose(state, [1, 2, 0])
        self.obs_dtype = state.dtype
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform an action in the environment and observe the result.

        Args:
            action (int): An action provided by the agent.

        Returns:
            observation (np.ndarray): The current state observation after taking the action.
            reward (float): The reward achieved by the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict[str, Any]): Additional information about the environment and episode.
        """
        action = self.available_actions[action]
        self.game.set_action(action)
        self.game.advance_action(self.frame_skip)

        state = self.game.get_state()
        reward = 0.0
        done = self.game.is_player_dead() or not state
        truncated = self.game.is_episode_finished()
        info = self.extra_statistics()

        observation = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.zeros(self.game_res, dtype=self.obs_dtype)
        if not done:
            self.game_variable_buffer.append(state.game_variables)

        self.store_statistics(self.game_variable_buffer)
        return observation, reward, done, truncated, info

    def get_available_actions(self) -> List[List[float]]:
        raise NotImplementedError

    def reward_wrappers_easy(self) -> List[gymnasium.RewardWrapper]:
        """
        Returns a list of reward wrapper classes for the dense reward setting.

        Returns:
            List[gymnasium.RewardWrapper]: A list of reward wrapper classes.
        """
        raise NotImplementedError

    def reward_wrappers_hard(self) -> List[gymnasium.RewardWrapper]:
        """
        Returns a list of reward wrapper classes for the sparse reward setting.

        Returns:
            List[gymnasium.RewardWrapper]: A list of reward wrapper classes.
        """
        raise NotImplementedError

    def get_and_update_user_var(self, game_var: GameVariable) -> int:
        """
        Retrieves and updates a user-defined variable from the game.

        Args:
            game_var (GameVariable): The game variable to retrieve and update.

        Returns:
            prev_var (int): The previous value of the specified game variable.
        """
        prev_var = self.user_variables[game_var]
        self.user_variables[game_var] = self.game.get_game_variable(game_var)
        return prev_var

    def render(self, mode="human"):
        """
        Renders the current state of the environment based on the specified mode.

        Args:
            mode (str): The mode for rendering (e.g., 'human', 'rgb_array').

        Returns:
            img (List[np.ndarray] or np.ndarray): Rendered image of the environment state.
        """
        state = self.game.get_state()
        img = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.uint8(np.zeros(self.game_res))
        if mode == 'human':
            if not self.render_enabled:
                return [img]
            try:
                # Render the image to the screen with swapped red and blue channels
                cv2.imshow('DOOM', img[:, :, [2, 1, 0]])
                cv2.waitKey(1)
            except Exception as e:
                print(f'Screen rendering unsuccessful: {e}')
                return np.zeros(img.shape)
        return [img]

    def clear_episode_statistics(self) -> None:
        """
        Clears or resets statistics collected during an episode.
        """
        self.game_variable_buffer.clear()

    def close(self):
        """
        Closes the Doom game instance.
        """
        self.game.close()
