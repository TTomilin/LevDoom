#!/usr/bin/env python
from __future__ import print_function

import logging
import os
from enum import Enum

import tensorflow as tf
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.cmd_util import make_vec_env

from callbacks import RecorderCallback
from config import argparser
from env import DefendTheCenter, HealthGathering, SeekAndKill, DodgeProjectiles
from util import get_input_shape


class Algorithm(Enum):
    PPO = PPO
    DQN = DQN


class Scenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenter
    HEALTH_GATHERING = HealthGathering
    SEEK_AND_KILL = SeekAndKill
    DODGE_PROJECTILES = DodgeProjectiles


if __name__ == "__main__":
    args = argparser()
    logging.getLogger().setLevel(logging.INFO)
    logging.info('Running GVizDoom with the following configuration')
    for key, val in args.__dict__.items():
        logging.info(f'{key}: {val}')

    # Configure on-demand memory allocation during runtime
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    logging.info(f'Num GPUs Available: {len(gpu_devices)}')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Find the root directory
    root_dir = os.path.dirname(os.getcwd())
    if 'GVizDoom' not in root_dir:
        root_dir += '/GVizDoom'  # Fix for external script executor

    # Determine the model
    if args.algorithm.upper() not in Algorithm._member_names_:
        raise ValueError(f'Unknown algorithm provided: `{args.algorithm}`')
    alg_name = args.algorithm.lower()
    algorithm = Algorithm[alg_name.upper()].value

    # Determine the scenario
    if args.scenario.upper() not in Scenario._member_names_:
        raise ValueError(f'Unknown scenario provided: `{args.scenario}`')
    scenario_class = Scenario[args.scenario.upper()].value

    # Instantiate a scenario for every task
    n_tasks = len(args.tasks)
    envs = [scenario_class(root_dir=root_dir, task=task, args=args) for task in args.tasks]
    eval_env = scenario_class(root_dir=root_dir, task=args.eval_task, args=args)

    # Validate input tasks
    env = envs[0]
    for task in args.tasks:
        if task.upper() not in env.task_list:
            raise ValueError(f'Unknown task provided for scenario {env.name}: `{task}`')

    extra_stats = env.extra_statistics

    # Define the state and action space
    action_size = env.game.get_available_buttons_size()
    state_size = get_input_shape(alg_name, args.frame_width, args.frame_height)

    # Create game instances for every task
    obs = env.reset()

    # env = make_vec_env(lambda: env, n_envs=n_envs, monitor_dir="./ppo_dtc_monitor", wrapper_class=ResizeObservation,
    #                    wrapper_kwargs={'shape': (args.frame_height, args.frame_width)})
    env = make_vec_env(lambda: env, n_envs=args.n_envs, monitor_dir="./ppo_dtc_monitor")
    eval_env = make_vec_env(lambda: eval_env)

    # Train the agent
    # model = PPO('MlpPolicy', env, verbose=2, device="cuda", tensorboard_log="./ppo_defend_the_center_tensorboard/")

    if args.load_model:
        model = algorithm.load("./models", env=env, device=args.device)
    else:
        model = algorithm('MultiInputPolicy', env, verbose=1, device=args.device, ent_coef=0.01,
                          tensorboard_log=f"./{args.algorithm}/{args.scenario}/")

    model = model.learn(total_timesteps=args.max_train_iterations, tb_log_name=f"{'_'.join(args.tasks)}_{args.log_name}",
                        eval_env=eval_env, eval_freq=1000, callback=RecorderCallback(env, eval_env, extra_stats))
    # model = model.learn(total_timesteps=100_000, tb_log_name="second_run", reset_num_timesteps=False)
    # Test the trained agent
    model.save("./models")
    obs = env.reset()
    n_episodes = 1000
    n_steps = 1000
    for eps in range(n_episodes):
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            if done.any():
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                logging.info("Done, episode=", eps)
                break
