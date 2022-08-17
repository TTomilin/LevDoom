from enum import Enum

from levdoom.algorithm.dqn import DQNImpl
from levdoom.algorithm.ppo import PPOImpl
from levdoom.algorithm.rainbow import RainbowImpl
from levdoom.env.base.defend_the_center import DefendTheCenter
from levdoom.env.base.dodge_projectiles import DodgeProjectiles
from levdoom.env.base.health_gathering import HealthGathering
from levdoom.env.base.seek_and_slay import SeekAndSlay
from levdoom.env.extended.defend_the_center_impl import DefendTheCenterImpl
from levdoom.env.extended.dodge_projectiles_impl import DodgeProjectilesImpl
from levdoom.env.extended.health_gathering_impl import HealthGatheringImpl
from levdoom.env.extended.seek_and_slay_impl import SeekAndSlayImpl


class DoomScenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenter
    HEALTH_GATHERING = HealthGathering
    SEEK_AND_SLAY = SeekAndSlay
    DODGE_PROJECTILES = DodgeProjectiles


class DoomScenarioImpl(Enum):
    DEFEND_THE_CENTER = DefendTheCenterImpl
    HEALTH_GATHERING = HealthGatheringImpl
    SEEK_AND_SLAY = SeekAndSlayImpl
    DODGE_PROJECTILES = DodgeProjectilesImpl


class Algorithm(Enum):
    DQN = DQNImpl
    PPO = PPOImpl
    RAINBOW = RainbowImpl
