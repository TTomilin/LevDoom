#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
from threading import Thread, Lock

from agent import DFPAgent, DuelingDDQNAgent, DRQNAgent
from doom import Doom
from memory import ExperienceReplay
from model import Algorithm, dueling_dqn, dueling_drqn, drqn, value_distribution_network, dfp_network, ModelVersion
from scenario import HealthGathering, SeekAndKill, DefendTheCenter, DodgeProjectiles
from trainer import AsynchronousTrainer
from utils import get_input_shape, next_model_version, latest_model_path

if __name__ == "__main__":

    # On-demand memory allocation during runtime
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Flags
    load_model = False
    load_experience = False
    save_experience = False
    prioritized_replay = True
    append_statistics = False
    decay_epsilon = True
    window_visible = False
    render_hud = True
    train = True

    # Config
    observe = 2000
    max_epochs = 3000
    learning_rate = 0.0001
    memory_storage_size = 5000
    memory_replay_capacity = 50000

    # Frequencies
    stats_save_freq = 5000
    model_save_freq = 5000
    memory_update_freq = 5000

    # Paths
    base_dir = '../../'
    model_name_addition = '_magic'

    # Specify model
    # algorithm = Algorithm.DFP
    # algorithm = Algorithm.DUELING_DDQN
    algorithm = Algorithm.DUELING_DDRQN
    # algorithm = Algorithm.C51_DDQN
    # algorithm = Algorithm.DRQN

    # Specify task(s)

    # current_tasks = [task for task in DefendTheCenter.SubTask]
    # current_tasks.remove(DefendTheCenter.SubTask.STONE_WALL)
    # current_tasks.remove(DefendTheCenter.SubTask.FLYING_ENEMIES)
    # current_tasks.remove(DefendTheCenter.SubTask.MULTI)
    # current_tasks = [task for task in HealthGathering.SubTask]
    # current_tasks.remove(HealthGathering.SubTask.STIMPACKS_POISON)
    # current_tasks.remove(HealthGathering.SubTask.SHADED_KITS)
    # current_tasks.remove(HealthGathering.SubTask.MULTI)
    # current_tasks = [task for task in DodgeProjectiles.SubTask]
    # current_tasks.remove(DodgeProjectiles.SubTask.MANCUBUS)
    # current_tasks.remove(DodgeProjectiles.SubTask.ARACHNOTRON)
    # current_tasks.remove(DodgeProjectiles.SubTask.MULTI)
    # current_tasks = [task for task in SeekAndKill.SubTask]
    # current_tasks.remove(SeekAndKill.SubTask.INVULNERABLE)
    # current_tasks.remove(SeekAndKill.SubTask.RED)
    # current_tasks.remove(SeekAndKill.SubTask.MULTI)

    # current_tasks = [SeekAndKill.SubTask.INVULNERABLE]
    # trained_task = SeekAndKill.SubTask.MULTI
    # current_tasks = [DodgeProjectiles.SubTask.MANCUBUS]
    # trained_task = DodgeProjectiles.SubTask.REVENANTS
    # current_tasks = [HealthGathering.SubTask.SUPREME]
    # trained_task = HealthGathering.SubTask.SUPREME

    # current_tasks = [HealthGathering.SubTask.STIMPACKS_POISON]
    current_tasks = [DefendTheCenter.SubTask.DEFAULT]
    trained_task = None

    # Create scenario(s)
    scenarios = [DefendTheCenter(base_dir, algorithm, task, trained_task, window_visible, len(current_tasks), render_hud, model_name_addition) for task in current_tasks]
    # scenarios = [HealthGathering(base_dir, algorithm, task, trained_task, window_visible, len(current_tasks), render_hud, model_name_addition) for task in current_tasks]
    # scenarios = [SeekAndKill(base_dir, algorithm, task, trained_task, window_visible, len(current_tasks), render_hud, model_name_addition) for task in current_tasks]
    # scenarios = [DodgeProjectiles(base_dir, algorithm, task, trained_task, window_visible, len(current_tasks), render_hud, model_name_addition) for task in current_tasks]

    # Define the dimensions
    # img_rows, img_cols = 64, 64
    img_rows, img_cols = 84, 84
    default_scenario = scenarios[0]
    action_size = default_scenario.game.get_available_buttons_size()
    state_size = get_input_shape(algorithm, img_rows, img_cols)

    # Build the model(s)
    if algorithm == Algorithm.DRQN:
        model = drqn(state_size, action_size, learning_rate)
        target_model = drqn(state_size, action_size, learning_rate)
    elif algorithm == Algorithm.DUELING_DDQN:
        model = dueling_dqn(state_size, action_size, learning_rate)
        target_model = dueling_dqn(state_size, action_size, learning_rate)
    elif algorithm == Algorithm.DUELING_DDRQN:
        model = dueling_drqn(state_size, action_size, learning_rate)
        target_model = dueling_drqn(state_size, action_size, learning_rate)
    elif algorithm == Algorithm.C51_DDQN:
        model = value_distribution_network(state_size, action_size, learning_rate)
        target_model = value_distribution_network(state_size, action_size, learning_rate)
    elif algorithm == Algorithm.DFP:
        model = dfp_network(state_size, action_size, learning_rate)
        target_model = None
    else:
        raise Exception(f'Algorithm not configured, received {algorithm}')

    # Use the name of the scenario if there are multiple tasks, otherwise decide based on training/testing
    task_name = default_scenario.name if len(current_tasks) > 1 else current_tasks[
        0].name.lower() if train else trained_task.name.lower()
    alg_name = algorithm.name.lower()

    model_path = f'{base_dir}models/{alg_name}/{default_scenario.name}/{task_name}{model_name_addition}_v*.h5'
    model_version = ModelVersion(next_model_version(model_path))

    # Load Model
    if load_model:
        path = latest_model_path(model_path)
        print(f'Loading model {path.split("/")[-1]}...')
        model.load_weights(path)
        if target_model:
            target_model.load_weights(path)

    # Create Memory
    memory_path = f'{base_dir}experience/{alg_name}_{default_scenario.name}_{default_scenario.task}.p'
    memory = ExperienceReplay(memory_path, save = save_experience, prioritized = prioritized_replay,
                              storage_size = memory_storage_size, capacity = memory_replay_capacity)

    # Create Agent
    # agent = DRQNAgent(memory, (img_rows, img_cols), state_size, action_size, model_path, model_version,
    agent = DuelingDDQNAgent(memory, (img_rows, img_cols), state_size, action_size, model_path, model_version,
    # agent = DFPAgent(memory, (img_rows, img_cols), state_size, action_size, model_path, model_version,
                     [model, target_model], Lock(), observe)

    # Create Trainer
    trainer = AsynchronousTrainer(agent, Lock(), decay_epsilon, model_save_freq = model_save_freq,
                                  memory_update_freq = memory_update_freq)

    # Load Experience
    if load_experience:
        agent.memory.load()

    games = [Doom(agent, scenario, stats_save_freq, max_epochs, append_statistics, train) for scenario in scenarios]

    # Play DOOM
    tasks = [Thread(target = doom.play) for doom in games]
    trainer_thread = Thread(target = trainer.train)

    for task in tasks:
        task.start()

    if train:
        trainer_thread.start()

    for task in tasks:
        task.join()

    if train:
        trainer_thread.join()

    # Close the game instance(s)
    for scenario in scenarios:
        scenario.game.close()
