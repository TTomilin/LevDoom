from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer

from levdoom.env.base.scenario import DoomEnv
from tianshou.data import ReplayBuffer, Collector
from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger


class BaseImpl(ABC):
    def __init__(self, args: Namespace, env: DoomEnv, log_path: str):
        self.args = args
        self.env = env
        self.log_path = log_path
        self.network = self.init_network()
        self.optimizer = self.init_optimizer()
        self.policy = self.init_policy()

    def get_policy(self):
        return self.policy

    @abstractmethod
    def create_trainer(self, train_collector: Collector, test_collector: Collector, logger: BaseLogger) -> Dict[str, Union[float, str]]:
        raise NotImplementedError

    @abstractmethod
    def create_buffer(self, n_buffers: int) -> ReplayBuffer:
        raise NotImplementedError

    @abstractmethod
    def init_network(self) -> Module:
        raise NotImplementedError

    @abstractmethod
    def init_optimizer(self) -> Optimizer:
        raise NotImplementedError

    @abstractmethod
    def init_policy(self) -> BasePolicy:
        raise NotImplementedError

    def save_best_fn(self, policy):
        torch.save(policy.state_dict(), f'{self.log_path}/policy_best.pth')

    def save_policy_fn(self, policy, env_step):
        torch.save(policy.state_dict(), f'{self.log_path}/policy_{env_step}.pth')

    def stop_fn(self, mean_rewards):
        return mean_rewards >= self.env.spec.reward_threshold if self.env.spec.reward_threshold else False
