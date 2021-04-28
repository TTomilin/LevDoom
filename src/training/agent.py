import copy
import itertools
import math
import random  # Handling random number generation
from threading import Lock
from typing import Tuple

import numpy as np  # Handle matrices
import skimage.color
import skimage.transform
from numpy import ndarray

from memory import ExperienceReplay
from model import ModelVersion


class Agent:
    """ Generic RL Agent
    Extend this abstract class to define new agents.
    """

    def __setstate__(self, state):
        print('Setting Agent state', state)
        self.__dict__.update(state)

    def __getstate__(self):
        state = self.__dict__.copy()
        print('Getting Agent state', state)
        return state

    def __init__(self,
                 memory: ExperienceReplay,
                 img_dims: tuple,
                 state_size: tuple,
                 action_size: int,
                 model_path: str,
                 model_version: ModelVersion,
                 models: [],
                 lock: Lock,
                 observe: int,
                 explore: int,
                 gamma: float,
                 batch_size: int,
                 frames_per_action: int,
                 update_target_freq: int,
                 ):

        # Dimensions
        self.img_dims = img_dims
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = batch_size

        # Training
        self.frames_per_action = frames_per_action
        self.observe = observe  # Collect Experience Replay
        self.explore = explore  # Decay epsilon
        self.time_step = None

        # Lock shared across threads
        self.lock = lock

        # Learning rate
        self.gamma = gamma
        self.epsilon = None

        # Frequencies
        self.update_target_freq = update_target_freq

        # Model
        self.model = models[0]
        self.target_model = models[1] if len(models) > 1 else None
        self.model_path = model_path
        self.model_version = model_version
        self.local_model_version = copy.deepcopy(model_version)

        # Memory
        self.memory = memory

    def transform_initial_state(self, game_state) -> []:
        """
        Implement this method to return the current state of the game
        :param game_state: state of the game
        :return: preprocessed and transformed initial state of the game
        """
        raise NotImplementedError

    def transform_new_state(self, old_state, new_state) -> []:
        """
        Implement this method to return the next state of the game
        :param old_state: old state of the game
        :param new_state: new state of the game
        :return: preprocessed and transformed new state of the game
        """
        raise NotImplementedError

    def get_action(self, state: dict, *args) -> int:
        """
        Implement this method to return the next best course
        of action for the agent based on its policy function
        :return: the index of the next best action
        """
        raise NotImplementedError

    def train(self) -> (float, float):
        """
        Implement this method to perform a weight update of the (online) network
        :return: None
        """
        raise NotImplementedError

    def update_target_model(self) -> None:
        """
        Copy the weights of the online network to the target network
        :return: None
        """
        print('Updating target model...')
        self.target_model.set_weights(self.model.get_weights())

    def preprocess_img(self, img, size = None):
        if size is None:
            size = self.img_dims
        img = np.rollaxis(img, 0, 3)  # (640, 480, 3)
        img = skimage.transform.resize(img, size)  # (64, 64, 3)
        return img

    def transform_state(self, game_state):
        return self.preprocess_img(game_state.screen_buffer)

    def load_model(self) -> None:
        """
        Load the weights from the specified path to both networks
        :return: None
        """
        path = self.latest_model_path()
        print(f"Loading model {path.split('/')[-1]}...")
        with self.lock:
            self.model.load_weights(path)
            self.update_target_model()

    def save_model(self) -> None:
        """
        Save the weights of the target network to the specified path
        Use local model version to prevent multiple threads saving the same model
        :return: None
        """
        from utils import next_model_version

        if self.local_model_version.version < self.model_version.version:
            self.local_model_version.version = self.model_version.version
            return
        with self.lock:  # Prevent other threads from using the model
            self.model_version.version = next_model_version(self.model_path)
            path = self.next_model_path()
            print(f"Saving model {path.split('/')[-1]}...")
            self.model.save_weights(path)

    def latest_model_path(self) -> str:
        return self.model_path.replace('*', str(self.model_version.version - 1))

    def next_model_path(self) -> str:
        return self.model_path.replace('*', str(self.model_version.version))


class DRQNAgent(Agent):

    def __init__(self,
                 memory: ExperienceReplay,
                 img_dims: tuple,
                 state_size: tuple,
                 action_size: int,
                 model_path: str,
                 model_version: ModelVersion,
                 models: [],
                 lock: Lock = None,
                 observe = 5000,
                 explore = 30000,
                 gamma = 0.99,
                 batch_size = 32,
                 trace_length = 4,
                 frames_per_action = 4,
                 update_target_freq = 3000
                 ):
        super().__init__(memory, img_dims, state_size, action_size, model_path, model_version, models, lock,
                         observe, explore, gamma, batch_size, frames_per_action, update_target_freq)
        self.trace_length = trace_length

    def transform_initial_state(self, game_state) -> []:
        return self.transform_state(game_state)

    def transform_new_state(self, _, new_state) -> []:
        return self.transform_state(new_state)

    def get_action(self, state: dict, *args) -> []:
        """
        Retrieve action using epsilon-greedy policy
        :param state: the current observable state that the agent is in
        :return: index of the action with the highest Q-value
        """
        if np.random.rand() <= self.epsilon or self.memory.buffer_size < self.trace_length:
            return random.randrange(self.action_size)
        with self.lock:
            last_traces = list(itertools.islice(self.memory.buffer, 0, self.trace_length))
            last_traces.reverse()
            last_traces = np.array([trace[3] for trace in last_traces])  # Take the next state of every trace
            last_traces = np.expand_dims(last_traces, axis = 0)  # 1x4x64x64x3
            q_values = self.target_model.predict(last_traces)
        return np.argmax(q_values)

    def train(self) -> None:
        """
        Train on [batch_size] random samples from experience replay
        :return: the maximum Q-value and the training loss
        """

        sample_traces = self.memory.sample(self.batch_size, self.trace_length)

        update_input = np.zeros(((self.batch_size,) + self.state_size))
        update_target = np.zeros(((self.batch_size,) + self.state_size))

        action = np.zeros((self.batch_size, self.trace_length))
        reward = np.zeros((self.batch_size, self.trace_length))

        for i in range(self.batch_size):
            for j in range(self.trace_length):
                update_input[i, j, :, :, :] = sample_traces[i][j][0]
                action[i, j] = sample_traces[i][j][1]
                reward[i, j] = sample_traces[i][j][2]
                update_target[i, j, :, :, :] = sample_traces[i][j][3]

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)

        for i in range(self.batch_size):
            a = np.argmax(target_val[i])
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])

        with self.lock:
            loss = self.model.train_on_batch(update_input, target)

        # Update the target model to be same with model
        if self.time_step % self.update_target_freq == 0:
            self.update_target_model()

        return np.max(target[-1, -1]), loss


class DuelingDDQNAgent(Agent):

    def __init__(self,
                 memory: ExperienceReplay,
                 img_dims: tuple,
                 state_size: tuple,
                 action_size: int,
                 model_path: str,
                 model_version: ModelVersion,
                 models: [],
                 lock: Lock = None,
                 observe = 5000,
                 explore = 30000,
                 gamma = 0.99,
                 batch_size = 32,
                 frames_per_action = 4,
                 update_target_freq = 3000
                 ):
        super().__init__(memory, img_dims, state_size, action_size, model_path, model_version, models, lock,
                         observe, explore, gamma, batch_size, frames_per_action, update_target_freq)

    def transform_initial_state(self, game_state: []) -> []:
        game_state = game_state.screen_buffer
        game_state = self.preprocess_img(game_state)
        game_state = np.stack(([game_state] * self.frames_per_action), axis = 2)  # 64x64x4
        game_state = np.expand_dims(game_state, axis = 0)  # 1x64x64x4
        return game_state

    def transform_new_state(self, old_state, new_state) -> []:
        width, height = self.img_dims
        new_state = new_state.screen_buffer
        new_state = self.preprocess_img(new_state)
        new_state = np.reshape(new_state, (1, width, height, 1))
        new_state = np.append(new_state, old_state[:, :, :, :3],
                              axis = 3)  # [x, x, x, x] -> [y, x, x, x] -> [z, y, x, x,]
        return new_state

    def preprocess_img(self, img, size = None):
        img = super().preprocess_img(img)
        img = skimage.color.rgb2gray(img)  # (64, 64)
        return img

    def get_action(self, state: dict, *args) -> int:
        """
        Retrieve action using epsilon-greedy policy
        :param state: the current observable state that the agent is in
        :return: index of the action with the highest Q-value
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with self.lock:
            q_values = self.target_model.predict(state)
        return np.argmax(q_values)

    def train(self) -> None:
        """
        Train on [batch_size] random samples from experience replay
        :param time_step: the total number of performed iterations
        :return: the maximum Q-value and the training loss
        """

        batch_size = self.batch_size if self.memory.prioritized else min(self.batch_size,
                                                                         self.memory.buffer_size)  # Smaller sample in case of insufficient experiences

        # Obtain random mini-batch from memory
        if self.memory.prioritized:
            tree_idx, mini_batch, ISWeights_mb = self.memory.sample(self.batch_size)
        else:
            mini_batch = self.memory.sample(batch_size)

        states = np.zeros(((batch_size,) + self.state_size))
        states_next = np.zeros(((batch_size,) + self.state_size))
        actions, rewards, done = [], [], []

        for i in range(self.batch_size):
            states[i, :, :, :] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            states_next[i, :, :, :] = mini_batch[i][3]
            done.append(mini_batch[i][-1])

        target = self.model.predict(states)
        target_val = self.target_model.predict(states_next)

        for i in range(batch_size):
            terminal = 0 if done[i] else 1  # Terminal state update only receives no future rewards
            best_action_value = np.max(
                target_val[i])  # Value of the best actions for the next state according to the target network
            target[i][actions[i]] = rewards[i] + terminal * self.gamma * best_action_value

        with self.lock:
            loss = self.model.train_on_batch(states, target)

        # Update priority
        if self.memory.prioritized:
            absolute_errors = loss  # ???
            self.memory.batch_update(tree_idx, absolute_errors)

        # Update the target model to be same with model
        if not self.time_step % self.update_target_freq:
            self.update_target_model()

        return np.max(target[-1, -1]), loss


class C51DDQNAgent(DuelingDDQNAgent):

    def __init__(self,
                 memory: ExperienceReplay,
                 img_dims: tuple,
                 state_size: tuple,
                 action_size: int,
                 model_path: str,
                 model_version: ModelVersion,
                 models: [],
                 lock: Lock = None,
                 observe = 5000,
                 explore = 30000,
                 gamma = 0.99,
                 batch_size = 32,
                 frames_per_action = 4,
                 update_target_freq = 3000,
                 num_atoms = 51,  # C51
                 v_max = 20.4,  # Max possible score for Defend the center is 26 - 0.1*26 - 0.3*10 = 20.4
                 v_min = -6.6,  # Min possible score for Defend the center is -0.1*26 - 1 - 0.3*10 = -6.6
                 ):
        super().__init__(memory, img_dims, state_size, action_size, model_path, model_version, models, lock,
                         observe, explore, gamma, batch_size, frames_per_action, update_target_freq)
        # Initialize Atoms
        self.num_atoms = num_atoms
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

    def get_action(self, state: [], *args) -> int:
        """
        Get action from model using epsilon-greedy policy
        """
        return random.randrange(self.action_size) if np.random.rand() <= self.epsilon else self.get_optimal_action(
            state)

    def get_optimal_action(self, state: []) -> []:
        """
        Get optimal action for a state
        """
        target = self.target_model.predict(state)  # Return a list [1x51, 1x51, 1x51]

        z_concat = np.vstack(target)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis = 1)

        # Pick action with the biggest Q value
        action_idx = np.argmax(q)

        return action_idx

    def train(self) -> Tuple[ndarray, float]:
        # Smaller sample size in case of insufficient experiences
        num_samples = self.batch_size if self.memory.prioritized else min(self.batch_size, self.memory.buffer_size)
        replay_samples = self.memory.sample(num_samples)

        state_inputs = np.zeros(((num_samples,) + self.state_size))
        next_states = np.zeros(((num_samples,) + self.state_size))
        m_prob = [np.zeros((num_samples, self.num_atoms)) for _ in range(self.action_size)]
        action, reward, done = [], [], []

        for i in range(num_samples):
            state_inputs[i, :, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            next_states[i, :, :, :] = replay_samples[i][3]
            done.append(replay_samples[i][-1])

        # Double DQN
        z = self.model.predict(next_states)  # Return a list [32x51, 32x51, 32x51]
        z_ = self.target_model.predict(next_states)  # Return a list [32x51, 32x51, 32x51]

        # Get Optimal Actions for the next states (from distribution z)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis = 1)  # length (num_atoms x num_actions)
        q = q.reshape((num_samples, self.action_size), order = 'F')
        optimal_action_idxs = np.argmax(q, axis = 1)

        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            if done[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, reward[i]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[action[i]][i][int(m_l)] += (m_u - bj)
                m_prob[action[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                    m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

        # Update the target model to be same with model
        if not self.time_step % self.update_target_freq:
            self.update_target_model()

        loss = self.model.fit(state_inputs, m_prob, batch_size = self.batch_size, epochs = 1, verbose = 0)
        return np.mean(q), loss.history['loss'][-1]


class DFPAgent(Agent):

    def __init__(self,
                 memory: ExperienceReplay,
                 img_dims: tuple,
                 state_size: tuple,
                 action_size: int,
                 model_path: str,
                 model_version: ModelVersion,
                 models: [],
                 lock: Lock = None,
                 observe = 5000,
                 explore = 30000,
                 gamma = 0.99,
                 batch_size = 32,
                 frames_per_action = 4,
                 update_target_freq = 3000,
                 measurement_size = 3,  # [Health, Medkit, Poison]
                 timesteps = [1, 2, 4, 8, 16, 32],
                 ):
        super().__init__(memory, img_dims, state_size, action_size, model_path, model_version, models, lock,
                         observe, explore, gamma, batch_size, frames_per_action, update_target_freq)
        n_timesteps = len(timesteps)
        self.measurement_size = measurement_size
        self.timesteps = timesteps
        self.goal_size = measurement_size * n_timesteps
        self.goal = np.array([1.0, 1.0, -1.0] * n_timesteps)

    def transform_initial_state(self, game_state: []) -> []:
        game_state = game_state.screen_buffer
        game_state = self.preprocess_img(game_state)
        game_state = np.stack(([game_state] * self.frames_per_action), axis = 2)  # 64x64x4
        game_state = np.expand_dims(game_state, axis = 0)  # 1x64x64x4
        return game_state

    def transform_new_state(self, old_state, new_state) -> []:
        width, height = self.img_dims
        new_state = new_state.screen_buffer
        new_state = self.preprocess_img(new_state)
        new_state = np.reshape(new_state, (1, width, height, 1))
        new_state = np.append(new_state, old_state[:, :, :, :3],
                              axis = 3)  # [x, x, x, x] -> [y, x, x, x] -> [z, y, x, x,]
        return new_state

    def preprocess_img(self, img, size = None):
        img = super().preprocess_img(img)
        img = skimage.color.rgb2gray(img)  # (64, 64)
        return img

    def get_action(self, state, *args):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        measurement = args[0]
        measurement = np.expand_dims(measurement, axis = 0)
        goal = np.expand_dims(self.goal, axis = 0)
        f = self.model.predict([state, measurement, goal])  # [1x6, 1x6, 1x6]
        f_pred = np.vstack(f)  # 3x6
        obj = np.sum(np.multiply(f_pred, self.goal), axis = 1)  # num_action

        return np.argmax(obj)

    # Pick samples randomly from replay memory (with batch_size)
    def train(self):
        """
        Train on a single mini-batch
        """
        batch_size = self.batch_size if self.memory.prioritized else min(self.batch_size, self.memory.buffer_size)
        rand_indices = np.random.choice(self.memory.buffer_size - (self.timesteps[-1] + 1), self.batch_size)

        state_input = np.zeros(((batch_size,) + self.state_size))  # Shape batch_size, img_rows, img_cols, 4
        measurement_input = np.zeros((batch_size, self.measurement_size))
        goal_input = np.tile(self.goal, (batch_size, 1))
        f_action_target = np.zeros((batch_size, (self.measurement_size * len(self.timesteps))))
        action = []

        for i, idx in enumerate(rand_indices):
            future_measurements = []
            last_offset = 0
            done = False
            for j in range(self.timesteps[-1] + 1):
                if not self.memory.buffer[idx + j][5]:  # if episode is not finished
                    if j in self.timesteps:  # 1,2,4,8,16,32
                        if not done:
                            future_measurements += list((self.memory.buffer[idx + j][4] - self.memory.buffer[idx][4]))
                            last_offset = j
                        else:
                            future_measurements += list((self.memory.buffer[idx + last_offset][4] - self.memory.buffer[idx][4]))
                else:
                    done = True
                    if j in self.timesteps:  # 1,2,4,8,16,32
                        future_measurements += list((self.memory.buffer[idx + last_offset][4] - self.memory.buffer[idx][4]))
            f_action_target[i, :] = np.array(future_measurements)
            state_input[i, :, :, :] = self.memory.buffer[idx][0]
            measurement_input[i, :] = self.memory.buffer[idx][4]
            action.append(self.memory.buffer[idx][1])

        f_target = self.model.predict([state_input, measurement_input, goal_input])  # Shape [32x18,32x18,32x18]

        for i in range(self.batch_size):
            f_target[action[i]][i, :] = f_action_target[i]

        loss = self.model.train_on_batch([state_input, measurement_input, goal_input], f_target)

        return np.nan, loss
