from argparse import Namespace
from typing import Dict, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

from levdoom.algorithm.base import BaseImpl
from levdoom.env.base.scenario import DoomEnv
from levdoom.network import DQN
from tianshou.data import ReplayBuffer, Collector, VectorReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BaseLogger


class DQNImpl(BaseImpl):
    def __init__(self, args: Namespace, env: DoomEnv, log_path: str):
        super(DQNImpl, self).__init__(args, env, log_path)

    def create_buffer(self, n_buffers: int) -> ReplayBuffer:
        return VectorReplayBuffer(
            self.args.buffer_size,
            buffer_num=n_buffers,
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=self.args.frames_stack
        )

    def create_trainer(self, train_collector: Collector, test_collector: Collector, logger: BaseLogger) -> Dict[
        str, Union[float, str]]:
        return offpolicy_trainer(
            self.policy,
            train_collector,
            test_collector,
            self.args.epoch,
            self.args.step_per_epoch,
            self.args.step_per_collect,
            self.args.test_num,
            self.args.batch_size,
            train_fn=self.train_fn,
            test_fn=self.test_fn,
            stop_fn=self.stop_fn,
            save_best_fn=self.save_best_fn,
            save_policy_fn=self.save_policy_fn,
            save_interval=self.args.save_interval,
            logger=logger,
            update_per_step=self.args.update_per_step,
            test_in_train=False
        )

    def init_network(self) -> Module:
        return DQN(*self.args.state_shape, self.args.action_shape, device=self.args.device)

    def init_optimizer(self) -> Optimizer:
        return torch.optim.Adam(self.network.parameters(), lr=self.args.lr)

    def init_policy(self) -> BasePolicy:
        return DQNPolicy(
            self.network,
            self.optimizer,
            self.args.gamma,
            self.args.n_step,
            target_update_freq=self.args.target_update_freq
        ).to(self.args.device)

    def train_fn(self, epoch, env_step):
        # Nature DQN setting, linear decay in the first 1M steps
        eps = self.args.eps_train - env_step / 1e6 * (
                    self.args.eps_train - self.args.eps_train_final) if env_step <= 1e6 else self.args.eps_train_final
        self.policy.set_eps(eps)

    def test_fn(self, epoch, env_step):
        self.policy.set_eps(self.args.eps_test)
