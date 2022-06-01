from argparse import Namespace

from torch.nn import Module

from levdoom.algorithm.dqn import DQNImpl
from levdoom.env.base.scenario import DoomEnv
from levdoom.network import Rainbow
from tianshou.data import ReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import BasePolicy, RainbowPolicy


class RainbowImpl(DQNImpl):
    def __init__(self, args: Namespace, env: DoomEnv, log_path: str):
        super(RainbowImpl, self).__init__(args, env, log_path)

    def create_buffer(self, n_buffers: int) -> ReplayBuffer:
        return PrioritizedVectorReplayBuffer(
            self.args.buffer_size,
            buffer_num=n_buffers,
            alpha=self.args.alpha,
            beta=self.args.beta,
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=self.args.frames_stack
        )

    def init_network(self) -> Module:
        return Rainbow(
            *self.args.state_shape,
            self.args.action_shape,
            self.args.num_atoms,
            device=self.args.device
        )

    def init_policy(self) -> BasePolicy:
        return RainbowPolicy(
            self.network,
            self.optimizer,
            self.args.gamma,
            self.args.num_atoms,
            self.args.v_min,
            self.args.v_max,
            self.args.n_step,
            target_update_freq=self.args.target_update_freq
        ).to(self.args.device)
