from collections import OrderedDict, deque
from enum import Enum

import numpy as np
import skimage
import tensorflow as tf
import time
from tensorforce import Runner
from tensorforce.core import TensorSpec
from tensorforce.environments import ViZDoom

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class VizDoomEnv(ViZDoom):
    def __init__(
            self, level, visualize = False, frame_skip = 4, seed = None
    ):
        super().__init__(level, visualize, True, True, frame_skip, seed)

        from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

        self.config_file = level
        self.visualize = visualize
        self.frame_skip = frame_skip
        self.current_states = None

        # Statistics
        self.total_reward = 0.0
        self.frames_alive = 0
        self.time_step = 0
        self.games = 0
        self.start_time = 0.0
        self.game_variables = deque(maxlen = 5)

        self.environment = DoomGame()
        self.environment.load_config(self.config_file)
        self.environment.set_mode(Mode.PLAYER)
        # self.environment.set_mode(Mode.ASYNC_PLAYER)
        self.environment.set_window_visible(self.visualize)
        # self.environment.set_screen_format(ScreenFormat.RGB24)
        self.environment.set_screen_format(ScreenFormat.CRCGCB)
        self.environment.set_screen_resolution(ScreenResolution.RES_640X480)
        self.environment.set_depth_buffer_enabled(False)
        self.environment.set_labels_buffer_enabled(False)
        self.environment.set_automap_buffer_enabled(False)
        if seed is not None:
            self.environment.setSeed(seed)
        self.environment.init()

        # self.state_shape = (64, 64, 4)
        # self.state_shape = (3, 480, 640)
        self.state_shape = (64, 64)
        self.num_variables = self.environment.get_available_game_variables_size()
        self.num_buttons = self.environment.get_available_buttons_size()

    def __str__(self):
        return super().__str__() + '({})'.format(self.config_file)

    def states(self):
        return OrderedDict(
            screen = dict(type = 'float', shape = self.state_shape, min_value = 0.0, max_value = 1.0),
            variables = dict(type = 'float', shape = self.num_variables))

    def actions(self):
        return dict(type = 'int', shape = (), num_values = self.num_buttons)

    def close(self):
        self.environment.close()
        self.environment = None

    def get_states(self):
        state = self.environment.get_state()
        # stacked_frames = transform_new_state(self.current_states['screen'], state) if self.current_states else transform_initial_state(state)
        stacked_frames = preprocess_img(state.screen_buffer, (64, 64))
        # return OrderedDict(screen = state.screen_buffer, variables = state.game_variables)
        return OrderedDict(screen = stacked_frames, variables = state.game_variables)

    def reset(self):
        self.start_time = time.time()
        self.environment.new_episode()
        self.current_states = self.get_states()
        return self.current_states

    def execute(self, actions):
        action = idx_to_action(actions, self.num_buttons)
        if self.visualize:
            self.environment.set_action(action)
            reward = 0.0
            for _ in range(self.frame_skip):
                self.environment.advance_action()
                reward += self.environment.get_last_reward()
        else:
            reward = self.environment.make_action(list(action), self.frame_skip)
        terminal = self.environment.is_episode_finished()
        self.game_variables.append(self.current_states['variables'])
        reward = shape_reward(reward, self.game_variables)
        self.total_reward += reward
        self.frames_alive += 1
        if not terminal:
            self.current_states = self.get_states()
        else:
            self.games += 1
            self.time_step += self.frames_alive
            duration = time.time() - self.start_time
            print(f'Episode Finished / games {self.games:d} / FRAMES {self.time_step:d} / TIME {duration:.2f} / '
                  f'REWARD {self.total_reward:.1f} / FRAMES SURVIVED {self.frames_alive:d} / TASK default')
            self.total_reward = 0.0
            self.frames_alive = 0
        return self.current_states, terminal, reward


# environment = VizDoomEnv(level = '../../scenarios/defend_the_center/defend_the_center.cfg',
environment = VizDoomEnv(level = '../../scenarios/dodge_projectiles/dodge_projectiles.cfg',
                         visualize = True, frame_skip = 4)


def preprocess_img(img, size):
    img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)
    return img


def idx_to_action(action_idx: int, action_size: int) -> []:
    actions = np.zeros([action_size])
    actions[action_idx] = 1
    actions = actions.astype(int)
    return actions.tolist()


def transform_initial_state(game_state: []) -> []:
    game_state = game_state.screen_buffer
    game_state = preprocess_img(game_state, (64, 64))
    game_state = np.stack(([game_state] * 4), axis = 2)  # 64x64x4
    # game_state = np.expand_dims(game_state, axis = 0)  # 1x64x64x4
    return game_state


# def shape_reward(reward: float, game_variables: deque) -> float:
#     if len(game_variables) < 2:
#         return reward
#     current_vars = game_variables[-1]
#     previous_vars = game_variables[-2]
#     if current_vars[DTCGameVariable.AMMO2.value] < previous_vars[DTCGameVariable.AMMO2.value]:
#         reward -= 0.1  # Use of ammunition
#     if current_vars[DTCGameVariable.HEALTH.value] < previous_vars[DTCGameVariable.HEALTH.value]:
#         reward -= 0.3  # Loss of HEALTH
#     return reward


def shape_reward(reward: float, game_variables: deque) -> float:
    if len(game_variables) < 2:
        return reward
    # +0.01 living reward already configured
    current_vars = game_variables[-1]
    previous_vars = game_variables[-2]
    if current_vars[DPGameVariable.HEALTH.value] < previous_vars[DPGameVariable.HEALTH.value]:
        reward -= 1  # Loss of HEALTH
    return reward


class DTCGameVariable(Enum):
    KILL_COUNT = 0
    AMMO2 = 1
    HEALTH = 2


class DPGameVariable(Enum):
    HEALTH = 0


def transform_new_state(old_state, new_state) -> []:
    new_state = new_state.screen_buffer
    new_state = preprocess_img(new_state, (64, 64))
    new_state = np.reshape(new_state, (64, 64, 1))
    new_state = np.append(new_state, old_state[:, :, :3],
                          axis = 2)  # [x, x, x, x] -> [y, x, x, x] -> [z, y, x, x,]
    return new_state


agent = dict(
    agent = 'ppo',
    # Automatically configured network
    network = 'auto',
    # PPO optimization parameters
    batch_size = 32, update_frequency = 2, learning_rate = 3e-4, multi_step = 10,
    subsampling_fraction = 0.33,
    # Reward estimation
    likelihood_ratio_clipping = 0.2, discount = 0.99, predict_terminal_values = False,
    # Baseline network and optimizer
    baseline = dict(type = 'auto', size = 32, depth = 1),
    baseline_optimizer = dict(optimizer = 'adam', learning_rate = 1e-3, multi_step = 10),
    # Regularization
    l2_regularization = 0.0, entropy_regularization = 0.0,
    # Preprocessing
    state_preprocessing = 'linear_normalization', reward_preprocessing = None,
    # Exploration
    exploration = 0.0, variable_noise = 0.0,
    # Default additional config values
    config = None,
    # Save agent every 10 updates and keep the 5 most recent checkpoints
    saver = dict(directory = 'model', frequency = 10, max_checkpoints = 5),
    # Log all available Tensorboard summaries
    summarizer = dict(directory = 'summaries', summaries = 'all'),
    # Do not record agent-environment interaction trace
    recorder = None
)

input_spec = TensorSpec(type = 'float', shape = (3, 480, 640), overwrite = True)

exploration = dict(
    type = 'linear', unit = 'timesteps', num_steps = 50000,
    initial_value = 1.0, final_value = 0.001
)

agent = dict(agent = 'dueling_dqn', memory = 20000, batch_size = 32, discount = 0.99, learning_rate = 1e-4,
             huber_loss = 0.9, target_sync_frequency = 3000, exploration = exploration,
             # state_preprocessing = [
             #     dict(type = 'image', width = 64, height = 64, grayscale = True, input_spec = input_spec),
             #     dict(type = 'linear_normalization', min_value = 0.0, max_value = 1.0)],
             # Save agent every 10 updates and keep the 5 most recent checkpoints
             saver = dict(directory = 'model', frequency = 10, max_checkpoints = 5),
             # Log all available Tensorboard summaries
             summarizer = dict(directory = 'summaries', summaries = 'all')
             )

# Initialize the runner
runner = Runner(agent = agent, environment = environment, max_episode_timesteps = 2100)

# Train for 3000 episodes
runner.run(num_episodes = 3000)
runner.close()
