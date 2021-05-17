#!/usr/bin/env python
from __future__ import print_function

import os
from enum import Enum, auto

from typing import List

import tensorflow as tf
from threading import Thread, Lock
import argparse

from agent import DFPAgent, DuelingDDQNAgent, DRQNAgent, C51DDQNAgent
from doom import Doom
from memory import ExperienceReplay
from scenario import HealthGathering, SeekAndKill, DefendTheCenter, DodgeProjectiles
from trainer import AsynchronousTrainer
from util import get_input_shape, next_model_version, latest_model_path


class Algorithm(Enum):
    DFP = DFPAgent
    DRQN = DRQNAgent
    C51_DDQN = C51DDQNAgent
    DUELING_DDQN = DuelingDDQNAgent


class Scenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenter
    HEALTH_GATHERING = HealthGathering
    SEEK_AND_KILL = SeekAndKill
    DODGE_PROJECTILES = DodgeProjectiles


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'GVizDoom runner')

    # Agent arguments
    parser.add_argument(
        '-a', '--algorithm', type = str, default = None, required = True,
        help = 'DRL algorithm used to construct the model [DQN, DRQN, Dueling_DDQN, DFP, C51] (case-insensitive)'
    )
    parser.add_argument(
        '--task-prioritization', type = bool, default = False,
        help = 'Scale the target weights according to task difficulty calculating the loss'
    )
    parser.add_argument(
        '--prioritized-replay', type = bool, default = False,
        help = 'Use PER (Prioritized Experience Reply) for storing and sampling the transitions'
    )
    parser.add_argument(
        '--decay-epsilon', type = str, default = True,
        help = 'Use epsilon decay for exploration'
    )
    parser.add_argument(
        '--train', type = str, default = True,
        help = 'Train or evaluate the agent'
    )
    parser.add_argument(
        '-o', '--observe', type = int, default = 10000,
        help = 'Number of iterations to collect experience before training'
    )
    parser.add_argument(
        '-e', '--explore', type = int, default = 50000,
        help = 'Number of iterations to decay the epsilon for'
    )
    parser.add_argument(
        '--learning-rate', type = float, default = 0.0001,
        help = 'Learning rate of the model optimizer'
    )
    parser.add_argument(
        '--KPI-update-frequency', type = int, default = 30,
        help = 'Number of episodes after which to update the Key Performance Indicator of a task for prioritization'
    )

    # Model arguments
    parser.add_argument(
        '--load-model', type = bool, default = False,
        help = 'Load existing weights or train from scratch'
    )
    parser.add_argument(
        '--model-save-frequency', type = int, default = 5000,
        help = 'Number of iterations after which to save a new version of the model'
    )
    parser.add_argument(
        '--model-name-addition', type = str, default = '',
        help = 'An additional identifier to the name of the model. Used to better differentiate stored data'
    )

    # Memory arguments
    parser.add_argument(
        '--load-experience', type = bool, default = False,
        help = 'Load existing experience into the replay buffer'
    )
    parser.add_argument(
        '--save-experience', type = bool, default = False,
        help = 'Store the gathered experience buffer externally'
    )
    parser.add_argument(
        '--memory-capacity', type = int, default = 50000,
        help = 'Number of most recent transitions to store in the replay buffer'
    )
    parser.add_argument(
        '--memory-storage-size', type = int, default = 5000,
        help = 'Number of most recent transitions to store externally from the replay buffer if saving is enabled'
    )
    parser.add_argument(
        '--memory-update_frequency', type = int, default = 5000,
        help = 'Number of iterations after which to externally store the most recent memory'
    )

    # Environment arguments
    parser.add_argument(
        '-s', '--scenario', type = str, default = None, required = True,
        help = 'Name of the scenario e.g., `defend_the_center` (case-insensitive)'
    )
    parser.add_argument(
        '-t', '--tasks', nargs="+", default = ['default'],
        help = 'List of tasks, e.g., `default gore stone_wall` (case-insensitive)'
    )
    parser.add_argument(
        '--trained-task', nargs="+", default = None,
        help = 'List of tasks, e.g., `default gore stone_wall`'
    )
    parser.add_argument(
        '-m', '--max-epochs', type = int, default = 3000,
        help = 'Maximum number of time steps per episode'
    )
    parser.add_argument(
        '-v', '--visualize', type = bool, default = False,
        help = 'Visualize the interaction of the agent with the environment'
    )
    parser.add_argument(
        '--render-hud', type = bool, default = True,
        help = 'Render the in-game hud, which displays health, ammo, armour, etc.'
    )
    parser.add_argument(
        '--frame-width', type = int, default = 84,
        help = 'Number of pixels to which the width of the frame is scaled down to'
    )
    parser.add_argument(
        '--frame-height', type = int, default = 84,
        help = 'Number of pixels to which the height of the frame is scaled down to'
    )

    # Statistics arguments
    parser.add_argument(
        '--append-statistics', type = str, default = True,
        help = 'Append the rolling statistics to an existing file or overwrite'
    )
    parser.add_argument(
        '--statistics-save-frequency', type = int, default = 5000,
        help = 'Number of iterations after which to write newly aggregated statistics'
    )

    # Parse the input arguments
    args = parser.parse_args()

    # Configure on-demand memory allocation during runtime
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f'Num GPUs Available: {gpu_devices}')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Find the root directory
    root_dir = os.path.dirname(os.getcwd())

    # Avoid deadlocks when operating on the networks
    lock = Lock()

    # Determine the model
    alg_name = args.algorithm.lower()
    algorithm = Algorithm[alg_name.upper()]

    # Find the scenario
    scenario_class = Scenario[args.scenario.upper()].value

    # Instantiate a scenario for every task
    n_tasks = len(args.tasks)
    scenarios = [scenario_class(root_dir, task, args.trained_task, args.visualize, n_tasks, args.render_hud,
                                args.model_name_addition) for task in args.tasks]

    # Validate input tasks
    scenario = scenarios[0]
    for task in args.tasks:
        if task not in scenario.task_list:
            raise ValueError(f'Unknown task provided for scenario {scenario.name}: `{task}`')

    # Define the state and action space
    action_size = scenario.game.get_available_buttons_size()
    state_size = get_input_shape(alg_name, args.frame_width, args.frame_height)

    # Set the task name as 'multi' if there are multiple tasks, otherwise decide based on training/testing
    task_name = 'multi' if n_tasks > 1 else args.tasks[0].lower() if args.train else args.trained_task.lower()
    model_path = f'{root_dir}/models/{alg_name}/{scenario.name}/{task_name}{args.model_name_addition}_v*.h5'

    # Instantiate Experience Replay
    memory_path_addition = scenario.task if n_tasks == 0 else 'shared'
    memory_path = f'{root_dir}/experience/{alg_name}_{scenario.name}_{memory_path_addition}.p'
    memory = ExperienceReplay(memory_path, save = args.save_experience, prioritized = args.prioritized_replay,
                              storage_size = args.memory_storage_size, capacity = args.memory_capacity)

    # Create Agent
    agent = algorithm.value(memory, (args.frame_width, args.frame_height), state_size, action_size, args.learning_rate,
                            model_path, lock, args.observe, task_prioritization = args.task_prioritization)

    # Load Model
    if args.load_model:
        agent.load_model()

    # Load Experience
    if args.load_experience:
        agent.memory.load()

    # Create Trainer
    trainer = AsynchronousTrainer(agent, args.decay_epsilon, model_save_freq = args.model_save_frequency,
                                  memory_update_freq = args.memory_update_frequency)

    # Create game instances for every task
    games = [Doom(agent, scenario, args.statistics_save_frequency, args.max_epochs, args.KPI_update_frequency,
                  args.append_statistics, args.train) for scenario in scenarios]

    # Play DOOM
    tasks = [Thread(target = doom.play) for doom in games]
    trainer_thread = Thread(target = trainer.train)

    # Run each task in a separate thread
    for task in tasks:
        task.start()

    # Run the asynchronous trainer
    if args.train:
        trainer_thread.start()

    # Terminate the trainer after the specified maximum training iterations are reached
    if args.train:
        trainer_thread.join()

    # Terminate the doom game instances after the specified maximum epochs are reached
    for task in tasks:
        task.join()

    # Close the game instance(s)
    for scenario in scenarios:
        scenario.game.close()
