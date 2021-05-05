#!/usr/bin/env python
from __future__ import print_function

import random
import time
import numpy as np
from collections import deque
from threading import get_ident

from agent import Agent
from scenario import Scenario
from stats_writer import Statistics
from utils import new_episode, idx_to_action


class Doom:
    def __init__(self, agent: Agent, scenario: Scenario, stats_save_freq = 5000, max_epochs = 3000,
                 append_statistics = True, train = True):
        self.train = train
        self.agent = agent
        self.scenario = scenario
        self.max_epochs = max_epochs
        self.stats_save_freq = stats_save_freq
        self.append_statistics = append_statistics

    def play(self) -> None:
        train = self.train
        agent = self.agent
        scenario = self.scenario
        print(f'Running scenario {scenario.task}, thread {get_ident()}')
    
        game = scenario.game
        game.init()
        game.new_episode()
        game_state = game.get_state()
        game_variables = deque(maxlen = scenario.len_vars_history)
        game_variables.append(game_state.game_variables)
    
        spawn_point_counter = {}
    
        # Create statistics manager
        statistics = Statistics(agent, scenario, self.stats_save_freq, self.append_statistics)
    
        # Statistic counters
        time_step, n_game, total_reward, frames_alive, total_duration = 0, 0, 0, 0, 0
        start_time = time.time()
    
        current_state = agent.transform_initial_state(game.get_state())

        # Number of health kits and poison pickup as measurements
        health_kits = 0
        poison = 0

        improved_epsilon = 1.0  # TODO remove after testing PER

        # Initial normalized measurements
        measurements = scenario.get_measurements(game_variables, health_kit = health_kits, poison = poison)

        while n_game < self.max_epochs:

            # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
            improved_epsilon = 0.01 + (1.0 - 0.01) * np.exp(-0.00005 * time_step)

            action_idx = agent.get_action(current_state, measurements)
            game.set_action(idx_to_action(action_idx, agent.action_size))
            game.advance_action(agent.frames_per_action)
            terminated = game.is_episode_finished()
            reward = game.get_last_reward()
    
            if terminated:
                n_game += 1
                duration = time.time() - start_time
                total_duration += duration
    
                # Append statistics
                statistics.append_episode(n_game, time_step, frames_alive, duration, total_reward, game_variables)
                # print(f'Improved epsilon {improved_epsilon:.4f}')
    
                # Reset counters
                frames_alive, total_reward = 0, 0
                start_time = time.time()
    
                # Restart
                game.set_seed(random.randint(0, 1000))
                new_episode(game, spawn_point_counter)
                game_variables.clear()
            else:
                frames_alive += 1
    
            new_state = game.get_state()
            game_variables.append(new_state.game_variables)
    
            new_state = agent.transform_new_state(current_state, new_state)

            if len(game_variables) > 1:
                current_vars = game_variables[-1]
                previous_vars = game_variables[-2]
                if previous_vars[0] - current_vars[0] > 8:  # Pick up Poison
                    poison += 1
                if current_vars[0] > previous_vars[0]:  # Pick up Health Pack
                    health_kits += 1
    
            reward = scenario.shape_reward(reward, game_variables)
            total_reward += reward
    
            # Save the sample <s, a, r, s', t> to the episode buffer
            if train:
                agent.memory.add((current_state, action_idx, reward, new_state, measurements, terminated))
                measurements = scenario.get_measurements(game_variables, health_kit = health_kits, poison = poison)
    
            current_state = new_state
            time_step += 1

            if terminated:
                health_kits = 0
                poison = 0
    
            # Save agent's performance statistics
            if not time_step % statistics.save_frequency:
                statistics.write(n_game, total_duration)
    
        # Final statistics
        statistics.write(n_game, total_duration)
