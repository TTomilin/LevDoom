from enum import Enum
from functools import partial
from typing import List, Callable

import gymnasium
from gymnasium import Env
from gymnasium.envs.registration import register, WrapperSpec
from gymnasium.wrappers import ResizeObservation, NormalizeObservation, FrameStackObservation

from levdoom.envs.base import DoomEnv
from levdoom.envs.defend_the_center.scenario import DefendTheCenter
from levdoom.envs.dodge_projectiles.scenario import DodgeProjectiles
from levdoom.envs.health_gathering.scenario import HealthGathering
from levdoom.envs.seek_and_slay.scenario import SeekAndSlay
from levdoom.utils.wrappers import RescaleObservation, RGBStack


class Scenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenter
    HEALTH_GATHERING = HealthGathering
    SEEK_AND_SLAY = SeekAndSlay
    DODGE_PROJECTILES = DodgeProjectiles


env_mapping = {
    Scenario.DEFEND_THE_CENTER: {
        'DefendTheCenterLevel0-v0': 'default',
        'DefendTheCenterLevel1_1-v0': 'gore',
        'DefendTheCenterLevel1_2-v0': 'mossy_bricks',
        'DefendTheCenterLevel1_3-v0': 'stone_wall',
        'DefendTheCenterLevel1_4-v0': 'fuzzy_enemies',
        'DefendTheCenterLevel1_5-v0': 'resized_enemies',
        'DefendTheCenterLevel1_6-v0': 'fast_enemies',
        'DefendTheCenterLevel1_7-v0': 'flying_enemies',
        'DefendTheCenterLevel2_1-v0': 'fast_flying_enemies',
        'DefendTheCenterLevel2_2-v0': 'gore_mossy_bricks',
        'DefendTheCenterLevel2_3-v0': 'resized_fuzzy_enemies',
        'DefendTheCenterLevel2_4-v0': 'stone_wall_flying_enemies',
        'DefendTheCenterLevel3_1-v0': 'resized_flying_enemies_mossy_bricks',
        'DefendTheCenterLevel3_2-v0': 'gore_stone_wall_fuzzy_enemies',
        'DefendTheCenterLevel3_3-v0': 'fast_resized_enemies_gore',
        'DefendTheCenterLevel4-v0': 'complete',
    },
    Scenario.HEALTH_GATHERING: {
        'HealthGatheringLevel0-v0': 'default',
        'HealthGatheringLevel1_1-v0': 'obstacles',
        'HealthGatheringLevel1_2-v0': 'slime',
        'HealthGatheringLevel1_3-v0': 'lava',
        'HealthGatheringLevel1_4-v0': 'water',
        'HealthGatheringLevel1_5-v0': 'stimpacks',
        'HealthGatheringLevel1_6-v0': 'supreme',
        'HealthGatheringLevel1_7-v0': 'resized_kits',
        'HealthGatheringLevel1_8-v0': 'short_agent',
        'HealthGatheringLevel1_9-v0': 'poison',
        'HealthGatheringLevel1_10-v0': 'shaded_kits',
        'HealthGatheringLevel2_1-v0': 'slime_obstacles',
        'HealthGatheringLevel2_2-v0': 'shaded_stimpacks',
        'HealthGatheringLevel2_3-v0': 'supreme_poison',
        'HealthGatheringLevel2_4-v0': 'resized_kits_lava',
        'HealthGatheringLevel2_5-v0': 'stimpacks_poison',
        'HealthGatheringLevel3_1-v0': 'lava_supreme_short_agent',
        'HealthGatheringLevel3_2-v0': 'obstacles_slime_stimpacks',
        'HealthGatheringLevel3_3-v0': 'poison_resized_shaded_kits',
        'HealthGatheringLevel4-v0': 'complete',
    },
    Scenario.SEEK_AND_SLAY: {
        'SeekAndSlayLevel0-v0': 'default',
        'SeekAndSlayLevel1_1-v0': 'blue',
        'SeekAndSlayLevel1_2-v0': 'red',
        'SeekAndSlayLevel1_3-v0': 'obstacles',
        'SeekAndSlayLevel1_4-v0': 'resized_enemies',
        'SeekAndSlayLevel1_5-v0': 'shadows',
        'SeekAndSlayLevel1_6-v0': 'mixed_enemies',
        'SeekAndSlayLevel1_7-v0': 'invulnerable',
        'SeekAndSlayLevel2_1-v0': 'blue_shadows',
        'SeekAndSlayLevel2_2-v0': 'obstacles_resized_enemies',
        'SeekAndSlayLevel2_3-v0': 'red_mixed_enemies',
        'SeekAndSlayLevel2_4-v0': 'invulnerable_blue',
        'SeekAndSlayLevel2_5-v0': 'resized_enemies_red',
        'SeekAndSlayLevel2_6-v0': 'shadows_obstacles',
        'SeekAndSlayLevel3_1-v0': 'blue_mixed_resized_enemies',
        'SeekAndSlayLevel3_2-v0': 'red_obstacles_invulnerable',
        'SeekAndSlayLevel3_3-v0': 'resized_shadows_invulnerable',
        'SeekAndSlayLevel4-v0': 'complete',
    },
    Scenario.DODGE_PROJECTILES: {
        'DodgeProjectilesLevel0-v0': 'default',
        'DodgeProjectilesLevel1_1-v0': 'city',
        'DodgeProjectilesLevel1_2-v0': 'flames',
        'DodgeProjectilesLevel1_3-v0': 'flaming_skulls',
        'DodgeProjectilesLevel1_4-v0': 'mancubus',
        'DodgeProjectilesLevel1_5-v0': 'barons',
        'DodgeProjectilesLevel1_6-v0': 'cacodemons',
        'DodgeProjectilesLevel1_7-v0': 'resized_agent',
        'DodgeProjectilesLevel2_1-v0': 'revenants',
        'DodgeProjectilesLevel2_2-v0': 'city_resized_agent',
        'DodgeProjectilesLevel2_3-v0': 'arachnotron',
        'DodgeProjectilesLevel2_4-v0': 'barons_flaming_skulls',
        'DodgeProjectilesLevel2_5-v0': 'cacodemons_flames',
        'DodgeProjectilesLevel2_6-v0': 'mancubus_resized_agent',
        'DodgeProjectilesLevel3_1-v0': 'flames_flaming_skulls_mancubus',
        'DodgeProjectilesLevel3_2-v0': 'resized_agent_revenants',
        'DodgeProjectilesLevel3_3-v0': 'city_arachnotron',
        'DodgeProjectilesLevel4-v0': 'complete',
    },
}


for scenario_enum in env_mapping:
    for env_name in env_mapping[scenario_enum]:
        register(
            id=env_name,
            entry_point=f'levdoom.envs.{scenario_enum.name.lower()}.scenario:{scenario_enum.value.__name__}',
            kwargs={'env': env_mapping[scenario_enum][env_name]},
        )


def wrap_env(env: Env, **kwargs):
    hard_rewards = kwargs.get('hard_rewards', False)
    reward_wrappers = env.unwrapped.reward_wrappers_hard() if hard_rewards else env.unwrapped.reward_wrappers_easy()
    for wrapper in reward_wrappers:
        env = wrapper.wrapper_class(env, **wrapper.kwargs)
    env = ResizeObservation(env, shape=(kwargs.get('frame_height', 84), kwargs.get('frame_width', 84)))
    if kwargs.get('rescale_observation', True):
        env = RescaleObservation(env)
    if kwargs.get('normalize_observation', True):
        env = NormalizeObservation(env)
    env = FrameStackObservation(env, kwargs.get('frame_stack', 4))
    env = RGBStack(env)
    return env


def make(env_id, **kwargs) -> Env:
    return wrap_env(gymnasium.make(env_id, **kwargs.get('doom', {})), **kwargs.get('wrap', {}))


def make_level(scenario: Scenario, level: int, **kwargs) -> List[Env]:
    env_ids = get_env_ids(level, scenario)
    return [make(env_id, **kwargs) for env_id in env_ids]


def make_level_fns(scenario: Scenario, level: int, **kwargs) -> List[Callable]:
    env_ids = get_env_ids(level, scenario)
    return [partial(make, env_id, **kwargs) for env_id in env_ids]


def get_env_ids(level, scenario):
    # Construct the key to match the scenario and level
    scenario_key = f"{scenario.value.__name__}Level{level}"
    return [env_id for env_id in env_mapping[scenario] if env_id.startswith(scenario_key)]
