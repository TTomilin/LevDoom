"""Policy package."""
# isort:skip_file

from tianshou.policy.base import BasePolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.policy.modelfree.c51 import C51Policy
from tianshou.policy.modelfree.rainbow import RainbowPolicy
from tianshou.policy.modelfree.pg import PGPolicy
from tianshou.policy.modelfree.a2c import A2CPolicy
from tianshou.policy.modelfree.ppo import PPOPolicy

__all__ = [
    "BasePolicy",
    "DQNPolicy",
    "C51Policy",
    "RainbowPolicy",
    "PGPolicy",
    "A2CPolicy",
    "PPOPolicy",
]
