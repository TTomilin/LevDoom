from enum import Enum

from levdoom.envs.defend_the_center.scenario import DefendTheCenter
from levdoom.envs.dodge_projectiles.scenario import DodgeProjectiles
from levdoom.envs.health_gathering.scenario import HealthGathering
from levdoom.envs.seek_and_slay.scenario import SeekAndSlay


class Scenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenter
    HEALTH_GATHERING = HealthGathering
    SEEK_AND_SLAY = SeekAndSlay
    DODGE_PROJECTILES = DodgeProjectiles
