import json
import os
from collections import deque
from enum import Enum, auto

import numpy as np

from agent import Agent
from scenario import Scenario


class State(Enum):
    OBSERVE = auto(),
    EXPLORE = auto(),
    TRAIN = auto(),
    TEST = auto()


class Statistics:

    def __init__(self, agent: Agent, scenario: Scenario, save_frequency = 25, append = True):
        self.agent = agent
        self.scenario = scenario
        self.save_frequency = save_frequency
        self.append = append
        self.buffers = {}
        for field in self.scenario.statistics_fields:
            self.buffers[field] = []

    def get_state(self, time_step: int) -> State:
        if self.scenario.trained_task:
            return State.TEST.name.lower()
        if time_step <= self.agent.observe:
            return State.OBSERVE.name.lower()
        if time_step <= self.agent.observe + self.agent.explore:
            return State.EXPLORE.name.lower()
        return State.TRAIN.name.lower()

    def log_episode(self, n_game: int, time_step: int, duration: float, total_reward: float, frames_alive: int) -> None:
        print(f'Episode Finished / n_game {n_game:d} / FRAMES {time_step:d} / TIME {duration:.2f} / '
              f'STATE {self.get_state(time_step):s} / EPSILON {self.agent.epsilon:.4f} / REWARD {total_reward:.1f} / '
              f'FRAMES SURVIVED {frames_alive:d} / TASK {self.scenario.task:s}')

    def append_episode(self, n_game: int, time_step: int, frames_alive: int, duration: float, reward: float,
                       game_vars: deque) -> None:

        # Info Log
        self.log_episode(n_game, time_step, duration, reward, frames_alive)
        self.buffers['frames_alive'].append(frames_alive)
        self.buffers['duration'].append(duration)
        self.buffers['reward'].append(reward)
        for key, value in self.scenario.additional_stats(game_vars).items():
            self.buffers[key].append(value)

    def get_statistics(self, n_game: int, total_duration: float):
        stats = {'episodes': n_game, 'total_duration': total_duration}
        for key, value in self.buffers.items():
            stats[key] = array_mean(self.buffers[key])
        return stats

    def write(self, n_game: int, total_duration: float) -> None:
        new_stats = self.get_statistics(n_game, float(np.around(total_duration, decimals = 2)))
        path = self.scenario.stats_path
        print(f'Updating Statistics {path.split("/")[-1]} - {new_stats}')
        stats_exist = os.path.exists(path)
        # print(f'Append: {append}, Stats exist: {stats_exist}')
        io_mode = 'r+' if stats_exist and self.append else 'w'
        with open(path, io_mode) as stats_file:
            stats = join_stats(stats_file, new_stats, ['episodes', 'total_duration']) \
                if stats_exist and self.append else new_stats
            # print(f'Full Statistic: {stats}')
            stats_file.seek(0)
            json.dump(stats, stats_file, indent = 4)
        self.append = True
        self.clear()

    def clear(self):
        for key in self.buffers.keys():
            self.buffers[key].clear()


def array_mean(values):
    return [np.around(np.mean(np.array(values), dtype = "float64"), decimals = 4)]


def join_stats(stats_file: str, new_stats: dict, override_keys: []) -> dict:
    return {key: new_stats[key] if key in override_keys else value + new_stats[key]
            for key, value in json.load(stats_file).items()}
