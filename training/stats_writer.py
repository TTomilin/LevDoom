from collections import deque
from enum import Enum, auto

import json
import numpy as np
import os

from .agent import Agent
from .scenario import Scenario
from .util import array_mean, join_stats, ensure_directory

MIN_KPI = 0.1


class State(Enum):
    OBSERVE = auto(),
    EXPLORE = auto(),
    TRAIN = auto(),
    TEST = auto()


class Statistics:

    def __init__(self, agent: Agent, scenario: Scenario, save_frequency, KPI_update_freq, append):
        self.agent = agent
        self.scenario = scenario
        self.save_frequency = save_frequency
        self.append = append
        self.KPI_update_freq = KPI_update_freq
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
              f'STATE {self.get_state(time_step):s} / EPSILON {self.agent.epsilon:.3f} / REWARD {total_reward:.1f} / '
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

        # Update KPI
        if not n_game % self.KPI_update_freq or n_game == 1:
            task_id = self.scenario.task
            KPI_type = self.scenario.get_performance_indicator()
            if KPI_type == Scenario.PerformanceIndicator.FRAMES_ALIVE:
                KPI = array_mean(self.buffers['frames_alive'])
            elif KPI_type == Scenario.PerformanceIndicator.KILL_COUNT:
                KPI = array_mean(self.buffers['kill_count'])
            else:
                print(f'KPI type {KPI_type} unimplemented')
                KPI = MIN_KPI
            self.agent.update_KPI(task_id, max(KPI, MIN_KPI))

    def get_statistics(self, n_game: int, total_duration: float):
        stats = {'episodes': n_game, 'total_duration': total_duration}
        for key, value in self.buffers.items():
            stats[key] = [array_mean(self.buffers[key])]
        return stats

    def write(self, n_game: int, total_duration: float) -> None:
        new_stats = self.get_statistics(n_game, float(np.around(total_duration, decimals = 2)))
        file_path = self.scenario.stats_path
        print(f'Updating Statistics {file_path.split("/")[-1]} - {new_stats}')
        stats_exist = os.path.exists(file_path)
        # print(f'Append: {append}, Stats exist: {stats_exist}')
        io_mode = 'r+' if stats_exist and self.append else 'w'
        ensure_directory(file_path)

        with open(file_path, io_mode) as stats_file:
            stats = join_stats(stats_file, new_stats, ['episodes', 'total_duration']) \
                if stats_exist and self.append else new_stats
            # print(f'Full Statistic: {stats}')
            stats_file.seek(0)
            json.dump(stats, stats_file, indent = 4)
        self.append = True
        self.clear_buffers()

    def clear_buffers(self):
        for key in self.buffers.keys():
            self.buffers[key].clear()
