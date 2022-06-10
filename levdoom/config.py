import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LevDoom runner')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['defend_the_center', 'health_gathering', 'seek_and_slay', 'dodge_projectiles'])
    parser.add_argument('--algorithm', type=str, default=None, choices=['dqn', 'ppo', 'rainbow'])
    parser.add_argument('--tasks', type=str, nargs='*', default=['default'])
    parser.add_argument('--test_tasks', type=str, nargs='*', default=['default'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=1e5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.)
    parser.add_argument('--v-max', type=float, default=10.)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--step-per-epoch', type=int, default=1e5)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--repeat-per-collect', type=int, default=4)
    parser.add_argument('--update-per-step', type=float, default=0.1)
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
    parser.add_argument('--with_wandb', default=False, type=bool, help='Enables Weights and Biases')
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
