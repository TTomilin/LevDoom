from argparse import Namespace
from typing import Dict, Union

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from levdoom.algorithm.base import BaseImpl
from levdoom.env.base.scenario import DoomEnv
from levdoom.network import DQN
from tianshou.data import VectorReplayBuffer, ReplayBuffer, Collector
from tianshou.policy import PPOPolicy, BasePolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import BaseLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic


class PPOImpl(BaseImpl):
    def __init__(self, args: Namespace, env: DoomEnv, log_path: str):
        super(PPOImpl, self).__init__(args, env, log_path)

    def create_buffer(self, n_buffers: int) -> ReplayBuffer:
        return VectorReplayBuffer(
            self.args.buffer_size,
            buffer_num=n_buffers,
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=self.args.frames_stack)

    def create_trainer(self, train_collector: Collector, test_collector: Collector, logger: BaseLogger) -> Dict[str, Union[float, str]]:
        return onpolicy_trainer(
            self.policy,
            train_collector,
            test_collector,
            self.args.epoch,
            self.args.step_per_epoch,
            self.args.repeat_per_collect,
            self.args.test_num,
            self.args.batch_size,
            step_per_collect=self.args.step_per_collect,
            stop_fn=self.stop_fn,
            save_best_fn=self.save_best_fn,
            save_policy_fn=self.save_policy_fn,
            save_interval=self.args.save_interval,
            logger=logger,
            test_in_train=False
        )

    def init_network(self) -> Module:
        network = DQN(*self.args.state_shape,
                   self.args.action_shape,
                   device=self.args.device,
                   features_only=True,
                   output_dim=self.args.hidden_size)
        self.actor = Actor(network, self.args.action_shape, device=self.args.device, softmax_output=False)
        self.critic = Critic(network, device=self.args.device)
        return network

    def init_optimizer(self) -> Optimizer:
        return torch.optim.Adam(ActorCritic(self.actor, self.critic).parameters(), lr=self.args.lr)

    def init_policy(self) -> BasePolicy:
        lr_scheduler = None
        if self.args.lr_decay:
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(
                self.args.step_per_epoch / self.args.step_per_collect
            ) * self.args.epoch

            lr_scheduler = LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1 - epoch / max_update_num
            )

        def dist(p):
            return torch.distributions.Categorical(logits=p)

        return PPOPolicy(
            self.actor,
            self.critic,
            self.optimizer,
            dist,
            discount_factor=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
            max_grad_norm=self.args.max_grad_norm,
            vf_coef=self.args.vf_coef,
            ent_coef=self.args.ent_coef,
            reward_normalization=self.args.rew_norm,
            action_scaling=False,
            lr_scheduler=lr_scheduler,
            action_space=self.env.action_space,
            eps_clip=self.args.eps_clip,
            value_clip=self.args.value_clip,
            dual_clip=self.args.dual_clip,
            advantage_normalization=self.args.norm_adv,
            recompute_advantage=self.args.recompute_adv
        ).to(self.args.device)
