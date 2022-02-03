#!/usr/bin/env python
from __future__ import print_function

import logging
import os
from enum import Enum

import tensorflow as tf
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from callbacks import RecorderCallback
from config import argparser
from env import DefendTheCenter, HealthGathering, SeekAndKill, DodgeProjectiles


class Algorithm(Enum):
    A2C = A2C
    PPO = PPO


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

    # Determine the algorithm
    if args.algorithm.upper() not in Algorithm._member_names_:
        raise ValueError(f'Unknown algorithm provided: `{args.algorithm}`')
    algorithm = Algorithm[args.algorithm.upper()].value

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

    extra_stats = env.statistics_fields

    # Create game instances for every task
    obs = env.reset()

    # TODO Extend for multiple environments
    env = make_vec_env(lambda: env, n_envs=args.n_envs, monitor_dir=f"{root_dir}/monitor")
    eval_env = make_vec_env(lambda: eval_env)

    # Create the agent
    if args.load_model:
        agent = algorithm.load(f"{root_dir}/models", env=env, device=args.device)
    else:
        agent = algorithm(ActorCriticCnnPolicy, env, verbose=1, device=args.device, ent_coef=args.ent_coef,
                          tensorboard_log=f"{root_dir}/logs/{args.scenario}/{args.algorithm}/")

    # Train the agent
    callback = RecorderCallback(env, eval_env, extra_stats, model_save_path=f'{root_dir}/models',
                                fps=args.render_fps, log_path=f'{root_dir}/logs')
    agent = agent.learn(total_timesteps=args.max_train_iterations, callback=callback,
                        tb_log_name=f"{'_'.join(args.tasks)}_{args.log_name}")

    # Save the model
    agent.save(f"{root_dir}/models")
