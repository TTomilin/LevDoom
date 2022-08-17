import os
import pathlib
import pprint
import sys
from argparse import Namespace
from datetime import datetime
from enum import Enum
from functools import partial
from typing import Type

import numpy as np
import torch
from gym.wrappers import NormalizeObservation, FrameStack
from tensorboardX import SummaryWriter

from levdoom.algorithm.dqn import DQNImpl
from levdoom.algorithm.ppo import PPOImpl
from levdoom.algorithm.rainbow import RainbowImpl
from levdoom.config import parse_args
from levdoom.env.base.scenario import DoomEnv
from levdoom.utils.wrappers import ResizeWrapper, RescaleWrapper

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))

from levdoom.env.extended.defend_the_center_impl import DefendTheCenterImpl
from levdoom.env.extended.dodge_projectiles_impl import DodgeProjectilesImpl
from levdoom.env.extended.health_gathering_impl import HealthGatheringImpl
from levdoom.env.extended.seek_and_slay_impl import SeekAndSlayImpl
from levdoom.utils.wandb import init_wandb
from tianshou.data.collector import Collector
from tianshou.env import ShmemVectorEnv
from tianshou.utils.logger.wandb import WandbLogger


class DoomScenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenterImpl
    HEALTH_GATHERING = HealthGatheringImpl
    SEEK_AND_SLAY = SeekAndSlayImpl
    DODGE_PROJECTILES = DodgeProjectilesImpl


class Algorithm(Enum):
    DQN = DQNImpl
    PPO = PPOImpl
    RAINBOW = RainbowImpl


def create_single_env(scenario: Type[DoomEnv], args: Namespace, task: str):
    env = scenario(args, task)
    env = ResizeWrapper(env, args.frame_height, args.frame_width)
    env = RescaleWrapper(env)
    env = NormalizeObservation(env)
    env = FrameStack(env, args.frame_stack)
    return env


def train(args: Namespace):
    args.tasks_joined = '_'.join(task for task in args.tasks)
    init_wandb(args)
    args.experiment_dir = pathlib.Path(__file__).parent.resolve()
    print('Experiment directory', args.experiment_dir)

    # Determine the log path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = f'{args.logdir}/{args.algorithm}/{args.scenario}/{args.seed}_{timestamp}'

    # Determine scenario and algorithm classes
    scenario_class = DoomScenario[args.scenario.upper()].value
    algorithm_class = Algorithm[args.algorithm.upper()].value

    args.cfg_path = f"{args.experiment_dir}/maps/{args.scenario}/{args.scenario}.cfg"
    args.res = (args.frame_skip, args.frame_height, args.frame_width)
    env = scenario_class(args, args.tasks[0])
    args.state_shape = args.res
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)  # should be N_FRAMES x H x W
    print("Actions shape:", args.action_shape)

    # Create training and testing environments
    train_envs = ShmemVectorEnv(
        [
            partial(create_single_env, scenario_class, args, task) for task in args.tasks
        ],
        norm_obs=args.normalize
    )
    test_envs = ShmemVectorEnv(
        [
            partial(create_single_env, scenario_class, args, task) for task in args.test_tasks
        ],
        norm_obs=args.normalize
    )

    # Apply the seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Initialize the algorithm
    algorithm = algorithm_class(args, env, log_path)

    policy = algorithm.get_policy()
    # Load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # Create replay buffer
    buffer = algorithm.create_buffer(len(train_envs))

    # Create collectors
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True, extra_statistics=env.extra_statistics)
    test_collector = Collector(policy, test_envs, exploration_noise=True, extra_statistics=env.extra_statistics)

    # Initialize logging
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    wandb_id = f'{args.algorithm}_seed_{args.seed}_{timestamp}'

    logger = WandbLogger(project=args.wandb_project,
                         name=wandb_id,
                         entity=args.wandb_user,
                         run_id=wandb_id,
                         config=args,
                         extra_statistics=env.extra_statistics)
    logger.load(writer)

    # Watch the agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = algorithm.create_buffer(len(test_envs))
            extra_statistics = ['kills', 'health', 'ammo', 'movement', 'kits_obtained', 'hits_taken']
            collector = Collector(policy, test_envs, buffer, exploration_noise=True, extra_statistics=extra_statistics)
            result = collector.collect(n_step=args.buffer_size, frame_skip=args.frame_skip)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.test_num, render=args.render_sleep, frame_skip=args.frame_skip)
        reward = result["reward"].mean()
        lengths = result["length"].mean() * args.frame_skip
        print(f'Mean reward (over {result["n/ep"]} episodes): {reward}')
        print(f'Mean length (over {result["n/ep"]} episodes): {lengths}')

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num, frame_skip=args.frame_skip)

    # Initialize trainer
    result = algorithm.create_trainer(train_collector, test_collector, logger)

    # Display the final results
    pprint.pprint(result)


if __name__ == '__main__':
    train(parse_args())
