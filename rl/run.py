import pathlib
import pprint
from argparse import Namespace
from datetime import datetime
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tianshou.data.collector import Collector
from tianshou.env import ShmemVectorEnv
from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.wandb import WandbLogger

import levdoom
from levdoom import Scenario
from rl.config import parse_args, Algorithm


def train(args: Namespace):
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = pathlib.Path(__file__).parent.parent.resolve()
    experiment_dir = pathlib.Path(__file__).parent.resolve()
    print('Experiment directory', experiment_dir)

    # Determine the log path
    scenario_name = args.scenario_name
    log_path = f'{args.logdir}/{args.algorithm}/{scenario_name}/{args.seed}_{args.timestamp}'

    # Determine the scenario and algorithm
    scenario = Scenario[scenario_name.upper()]
    algorithm_class = Algorithm[args.algorithm.upper()].value

    dummy_env = levdoom.make_level(scenario, 0)[0]
    args.state_shape = dummy_env.observation_space.shape
    args.action_shape = dummy_env.action_space.n
    print("Observations shape:", args.state_shape)  # should be N_FRAMES x H x W
    print("Actions shape:", args.action_shape)

    doom_kwargs = {
        'frame_skip': args.frame_skip,
        'seed': args.seed,
        'render': args.render,
        'resolution': args.resolution,
        'variable_queue_length': args.variable_queue_len,
    }

    wrapper_kwargs = {
        'frame_height': args.frame_height,
        'frame_width': args.frame_width,
        'frame_stack': args.frame_stack,
        'hard_rewards': args.hard_rewards,
    }

    kwargs = {
        'doom': doom_kwargs,
        'wrap': wrapper_kwargs,
    }

    # Create training and testing environments
    train_envs = create_vec_env(scenario, args.train_levels, **kwargs)
    test_envs = create_vec_env(scenario, args.test_levels, **kwargs)

    # Apply the seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Initialize the algorithm
    algorithm = algorithm_class(args, log_path)

    policy = algorithm.get_policy()
    # Load a previous policy
    if args.resume_path:
        resume_path = f'{args.logdir}/{args.algorithm}/{scenario_name}/{args.resume_path}'
        policy.load_state_dict(torch.load(resume_path, map_location=args.device))
        print("Loaded agent from: ", resume_path)

    # Create replay buffer
    buffer = algorithm.create_buffer(len(train_envs))

    # Create collectors
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # Initialize logging
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))

    wandb_id = f'{args.algorithm}_seed_{args.seed}_{args.timestamp}'

    logger = TensorboardLogger(writer) if not args.with_wandb else WandbLogger(project=args.wandb_project,
                                                                               name=wandb_id,
                                                                               entity=args.wandb_user,
                                                                               run_id=wandb_id,
                                                                               config=args)

    if args.with_wandb:
        logger.load(writer)

    train_collector.collect(n_step=args.batch_size * args.training_num)

    # Initialize trainer
    result = algorithm.create_trainer(train_collector, test_collector, logger)

    # Display the final results
    pprint.pprint(result)


def create_vec_env(scenario, levels, **kwargs):
    env_callables = []
    for level in levels:
        env_callables.extend(levdoom.make_level_fns(scenario, level, **kwargs))
    return ShmemVectorEnv(env_callables)


if __name__ == '__main__':
    train(parse_args())
