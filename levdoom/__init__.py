from functools import partial
from typing import List, Callable

import gymnasium
from gymnasium import Env
from gymnasium.envs.registration import register, WrapperSpec
from gymnasium.wrappers import ResizeObservation, NormalizeObservation, FrameStack

from levdoom.envs.base import DoomEnv
from levdoom.envs.config import env_mapping
from levdoom.utils.enums import Scenario
from levdoom.utils.wrappers import RescaleObservation, RGBStack

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
    env = RescaleObservation(env)
    env = NormalizeObservation(env)
    env = FrameStack(env, kwargs.get('frame_stack', 4))
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
