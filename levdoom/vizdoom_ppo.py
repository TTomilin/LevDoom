import argparse
import os
import pathlib
import pprint
import sys
from datetime import datetime
from enum import Enum
from functools import partial

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))

from levdoom.env.extended.defend_the_center_impl import DefendTheCenterImpl
from levdoom.env.extended.dodge_projectiles_impl import DodgeProjectilesImpl
from levdoom.env.extended.health_gathering_impl import HealthGatheringImpl
from levdoom.env.extended.seek_and_slay_impl import SeekAndSlayImpl
from levdoom.wandb_utils import init_wandb
from network import DQN
from tianshou.data import VectorReplayBuffer
from tianshou.data.collector import Collector
from tianshou.env import ShmemVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.logger.wandb import WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic


class DoomScenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenterImpl
    HEALTH_GATHERING = HealthGatheringImpl
    SEEK_AND_SLAY = SeekAndSlayImpl
    DODGE_PROJECTILES = DodgeProjectilesImpl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default=None)
    parser.add_argument('--tasks', type=str, nargs='*', default=['default'])
    parser.add_argument('--test_tasks', type=str, nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--repeat-per-collect', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--rew-norm', type=int, default=False)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render_sleep', type=float, default=0.)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--variable_queue_len', type=int, default=5)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frame-size', type=int, default=84)
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--skip-num', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    parser.add_argument(
        '--save-lmp',
        default=False,
        action='store_true',
        help='save lmp file for replay whole episode'
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)
    parser.add_argument(
        '--icm-lr-scale',
        type=float,
        default=0.,
        help='use intrinsic curiosity module with this lr scale'
    )
    parser.add_argument(
        '--icm-reward-scale',
        type=float,
        default=0.01,
        help='scaling factor for intrinsic curiosity reward'
    )
    parser.add_argument(
        '--icm-forward-loss-weight',
        type=float,
        default=0.2,
        help='weight for the forward model loss in ICM'
    )
    # WandB
    parser.add_argument('--with_wandb', default=True, type=bool, help='Enables Weights and Biases')
    parser.add_argument('--wandb_user', default=None, type=str, help='WandB username (entity).')
    parser.add_argument('--wandb_project', default='LevDoom', type=str, help='WandB "Project"')
    parser.add_argument('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
    parser.add_argument('--wandb_job_type', default='default', type=str, help='WandB job type')
    parser.add_argument('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb_dir', default=None, type=str, help='the place to save WandB files')
    # Scenario specific
    parser.add_argument('--kill_reward', default=1.0, type=float, help='For eliminating an enemy')
    parser.add_argument('--health_acquired_reward', default=1.0, type=float, help='For picking up health kits')
    parser.add_argument('--health_loss_penalty', default=0.1, type=float, help='Negative reward for losing health')
    parser.add_argument('--ammo_used_penalty', default=0.1, type=float, help='Negative reward for using ammo')
    parser.add_argument('--traversal_reward_scaler', default=1e-3, type=float,
                        help='Reward scaler for traversing the map')
    parser.add_argument('--add_speed', default=False, action='store_true')
    return parser.parse_args()


def train_ppo(args=get_args()):
    args.algorithm = 'ppo'
    args.tasks_joined = '_'.join(task for task in args.tasks)
    init_wandb(args)
    args.experiment_dir = pathlib.Path(__file__).parent.resolve()
    print('Experiment directory', args.experiment_dir)

    # Determine scenario class
    if args.scenario.upper() not in DoomScenario._member_names_:
        raise ValueError(f'Unknown scenario provided: `{args.scenario}`')
    scenario_class = DoomScenario[args.scenario.upper()].value

    args.cfg_path = f"{args.experiment_dir}/maps/{args.scenario}/{args.scenario}.cfg"
    args.res = (args.skip_num, args.frame_size, args.frame_size)
    env = scenario_class(args, args.tasks[0])
    args.state_shape = args.res
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observations shape:", args.state_shape)  # should be N_FRAMES x H x W
    print("Actions shape:", args.action_shape)
    # make environments
    train_envs = ShmemVectorEnv(
        [
            partial(scenario_class, args, task) for task in args.tasks
        ],
        norm_obs=args.normalize
    )
    test_envs = ShmemVectorEnv(
        [
            partial(scenario_class, args, task) for task in args.test_tasks
        ],
        norm_obs=args.normalize
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    net = DQN(
        *args.state_shape,
        args.action_shape,
        device=args.device,
        features_only=True,
        output_dim=args.hidden_size
    )
    actor = Actor(net, args.action_shape, device=args.device, softmax_output=False)
    critic = Critic(net, device=args.device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    # define policy
    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )
    extra_statistics = ['kills', 'health', 'ammo', 'movement', 'kits_obtained', 'hits_taken']
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True, extra_statistics=extra_statistics)
    test_collector = Collector(policy, test_envs, exploration_noise=True, extra_statistics=extra_statistics)
    # log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = f'{args.logdir}/{args.algorithm}/{args.scenario}/{args.seed}_{timestamp}'
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.scenario:
        args.wandb_group = args.scenario

    wandb_id = f'{args.algorithm}_seed_{args.seed}_{timestamp}'

    logger = WandbLogger(project=args.wandb_project,
                         name=wandb_id,
                         entity=args.wandb_user,
                         run_id=wandb_id,
                         config=args,
                         extra_statistics=extra_statistics)
    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), f'{log_path}/policy_best.pth')

    def save_policy_fn(policy, env_step):
        torch.save(policy.state_dict(), f'{log_path}/policy_{env_step}.pth')

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold if env.spec.reward_threshold else False

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True, extra_statistics=extra_statistics)
            result = collector.collect(n_step=args.buffer_size, frame_skip=args.skip_num)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.test_num, render=args.render_sleep, frame_skip=args.skip_num)
        rew = result["reward"].mean()
        lens = result["length"].mean() * args.skip_num
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
        print(f'Mean length (over {result["n/ep"]} episodes): {lens}')

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num, frame_skip=args.skip_num)
    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_policy_fn=save_policy_fn,
        save_interval=args.save_interval,
        logger=logger,
        test_in_train=False
    )

    pprint.pprint(result)


if __name__ == '__main__':
    train_ppo(get_args())
