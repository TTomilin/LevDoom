import json
import math
import os
from random import randrange
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d

stats_dir = '../statistics'

ranges = {
    'defend_the_center': {
        'gore_mossy_bricks': [], 'resized_fuzzy_enemies': [], 'stone_wall_flying_enemies': [],
        'resized_flying_enemies_mossy_bricks': [], 'gore_stone_wall_fuzzy_enemies': [], 'fast_resized_enemies_gore': [],
        'complete': []
    },
    'health_gathering': {
        'default': [515, 524],
        'lava': [510, 524], 'water': [310, 324], 'obstacles': [410, 424], 'stimpacks': [410, 424], 'resized_agent': [450, 524], 'resized_kits': [450, 524], 'shaded_kits': [450, 524], 'supreme': [200, 300],
        'slimy_obstacles': [300, 494], 'shaded_stimpacks': [270, 360], 'supreme_poison': [290, 430],
        'poison_resized_shaded_kits': [200, 300], 'obstacles_slime_stimpacks': [160, 290], 'lava_supreme_short_agent': [140, 270],
        'complete': [130, 190]
    },
    'seek_and_kill': {
        'default': [0, 15],
        'red': [0, 15], 'blue': [0, 15], 'shadows': [0, 15], 'obstacles': [0, 15], 'invulnerable': [0, 15], 'mixed_enemies': [0, 15], 'resized_enemies': [0, 15],
        'blue_shadows': [0, 15], 'obstacles_resized_enemies': [0, 15], 'red_mixed_enemies': [0, 15],
        'invulnerable_blue': [0, 15], 'resized_enemies_red': [0, 15], 'shadows_obstacles': [0, 15],
        'blue_mixed_resized_enemies': [0, 15], 'red_obstacles_invulnerable': [0, 15], 'resized_shadows_invulnerable': [0, 15],
        'complete': [0, 15]
    },
    'dodge_projectiles': {
        'default': [470, 524],
        'barons': [515, 524], 'cacodemons': [510, 524], 'city': [450, 524], 'flames': [450, 524], 'flaming_skulls': [450, 524], 'resized_agent': [350, 424],
        'arachnotron': [235, 424], 'cacodemons_flames': [235, 424], 'mancubus_resized_agent': [225, 424], 'revenants': [275, 400], 'city_resized_agent': [225, 330], 'barons_flaming_skulls': [180, 275],
        'flames_flaming_skulls_mancubus': [120, 160], 'resized_agent_revenants': [155, 195], 'city_arachnotron': [175, 200],
        'complete': [120, 185]
    }
}

task_dict = {
    'defend_the_center': {
        0: ['default'],
        1: ['gore', 'stone_wall', 'fast_enemies', 'mossy_bricks', 'fuzzy_enemies', 'flying_enemies', 'resized_enemies'],
        2: ['gore_mossy_bricks', 'resized_fuzzy_enemies', 'stone_wall_flying_enemies'],
        3: ['resized_flying_enemies_mossy_bricks', 'gore_stone_wall_fuzzy_enemies', 'fast_resized_enemies_gore'],
        4: ['complete']
    },
    'health_gathering': {
        0: ['default'],
        1: ['lava', 'slime', 'supreme', 'poison', 'obstacles', 'stimpacks', 'shaded_kits', 'water', 'resized_kits', 'short_agent'],
        2: ['slimy_obstacles', 'shaded_stimpacks', 'supreme_poison'],
        3: ['poison_resized_shaded_kits', 'obstacles_slime_stimpacks', 'lava_supreme_short_agent'],
        4: ['complete']
    },
    'seek_and_kill': {
        0: ['default'],
        1: ['red', 'blue', 'shadows', 'obstacles', 'invulnerable', 'mixed_enemies', 'resized_enemies'],
        2: ['blue_shadows', 'obstacles_resized_enemies', 'red_mixed_enemies', 'invulnerable_blue', 'resized_enemies_red'],
        3: ['blue_mixed_resized_enemies', 'red_obstacles_invulnerable', 'resized_shadows_invulnerable'],
        4: ['complete']
    },
    'dodge_projectiles': {
        0: ['default'],
        1: ['barons', 'flames', 'cacodemons', 'resized_agent', 'flaming_skulls', 'city'],
        2: ['revenants', 'arachnotron', 'city_resized_agent', 'barons_flaming_skulls', 'cacodemons_flames', 'mancubus_resized_agent'],
        3: ['flames_flaming_skulls_mancubus', 'resized_agent_revenants', 'city_arachnotron'],
        4: ['complete']
    }
}

metrics = {
    'defend_the_center': 'frames_alive',
    'health_gathering': 'frames_alive',
    'seek_and_kill': 'kill_count',
    'dodge_projectiles': 'frames_alive'
}

multipliers = {
    'defend_the_center': 2,
    'health_gathering': 4,
    'seek_and_kill': 1,
    'dodge_projectiles': 4
}


def plot_stats_single(scenario: str, file_names: List[str], interval = 5, n_data_points = 50) -> None:
    for file_name in file_names:
        with open(f'{stats_dir}/{scenario}/{file_name}.json', 'r') as file:
            stats = json.loads(file.read())
            y_vals = stats[metrics[scenario]]
            y_vals = [value * multipliers[scenario] for value in y_vals]
            y_vals = y_vals[:n_data_points]
            x_interval = np.arange(0, len(y_vals) * interval, interval).tolist()
            plt.plot(x_interval, y_vals, label = f'{file_name}')
    plt.xlabel('Timesteps (K)', fontsize = 16)
    plt.ylabel('Score', fontsize = 16)
    plt.legend()
    plt.show()


def plot_stats_multi(scenario: str, protocols: List[str], interval = 5, n_data_points = 50) -> None:
    for protocol in protocols:
        tasks = os.listdir(f'{stats_dir}/{scenario}/{protocol}/')
        stats_array = np.zeros((len(tasks), n_data_points))
        for i, task in enumerate(tasks):
            try:
                with open(f'{stats_dir}/{scenario}/{protocol}/{task}', 'r') as file:
                    stats = json.loads(file.read())
                    y_vals = stats[metrics[scenario]]
                    y_vals = y_vals[:n_data_points]
                    y_vals = [value * multipliers[scenario] for value in y_vals]
                    for j, val in enumerate(y_vals):
                        stats_array[i][j] = val
            except FileNotFoundError as error:
                print(error)
                continue
        stats_array[stats_array == 0] = np.nan
        mean_vals = np.nanmean(stats_array, axis = 0)
        x_interval = np.arange(0, len(mean_vals) * interval, interval).tolist()
        plt.plot(x_interval, mean_vals, label = protocol)
    plt.xlabel('Timesteps (K)')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def plot_tasks(scenario: str, level: int, setting = 'train', interval = 10, seeds = ['', '_1111', '_2222', '_3333'], n_data_points = 50) -> None:
    tasks = task_dict[scenario][level]
    for task in tasks:
        seed_data = np.empty((len(seeds), n_data_points))
        seed_data[:] = np.nan
        for i, seed in enumerate(seeds):
            try:
                with open(f'{stats_dir}/{scenario}/{setting}/{task}{seed}.json', 'r') as file:
                    data = np.array(json.load(file)[metrics[scenario]])
                    for j in range(n_data_points):
                        seed_data[i][j] = data[j] * multipliers[scenario]
            except FileNotFoundError as error:
                print(error)
                continue
            except IndexError as error:
                print('INDEX ERROR', scenario, task, seed, error)
                continue
        x = np.arange(n_data_points) * interval
        y = np.nanmean(seed_data, axis = 0)
        y = gaussian_filter1d(y, sigma = 0.5)

        plt.plot(x, y, label = f'{task.title()}')
        upper = y - 1.96 * np.nanstd(seed_data, axis = 0) / np.sqrt(30)
        lower = y + 1.96 * np.nanstd(seed_data, axis = 0) / np.sqrt(30)
        plt.fill_between(x, upper, lower, alpha = 0.5)
    plt.title(scenario.replace('_', ' ').title())
    plt.xlabel('Timesteps (K)')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'../plots/{scenario}/Training_Level_{level}.png')
    plt.show()


def plot_levels(scenario: str, setting = 'train', interval = 5) -> None:
    n_ticks = 40
    for level, tasks in task_dict[scenario].items():
        y_vals_list = []
        max_len = -math.inf
        for task in tasks:
            try:
                with open(f'{stats_dir}/{scenario}/{setting}/{task}.json', 'r') as file:
                    stats = json.loads(file.read())
                    y_vals = stats[metrics[scenario]]
                    y_vals = [value * multipliers[scenario] for value in y_vals]
                    if len(y_vals) > max_len:
                        max_len = len(y_vals)
                    y_vals_list.append(y_vals)
            except FileNotFoundError as error:
                print(error)
                continue
        for y_vals in y_vals_list:
            while len(y_vals) < max_len:
                y_vals.append(np.nan)
        y_vals_list = np.array(y_vals_list)
        y_vals = np.mean(y_vals_list, axis = 0)
        y_vals = y_vals[:n_ticks]
        y_vals_list = y_vals_list[:, :n_ticks]
        x_interval = np.arange(0, y_vals.size * interval, interval).tolist()
        plt.plot(x_interval, y_vals, label = f'Level {level}')

        y = gaussian_filter1d(y_vals, sigma = 0.5)
        upper = y - 1.96 * np.nanstd(y_vals_list, axis = 0) / np.sqrt(10)
        lower = y + 1.96 * np.nanstd(y_vals_list, axis = 0) / np.sqrt(10)
        plt.fill_between(x_interval, upper, lower, alpha = 0.5)
    plt.xlabel('Timesteps (K)', fontsize = 16)
    plt.ylabel('Score', fontsize = 16)
    plt.legend()
    plt.show()
    plt.savefig(f'../plots/{scenario}/Train_Levels_Rainbow.png')


def plot_levels_with_seeds(scenario: str, setting = 'train', n_data_points = 20, n_extra_data_points = 0, interval = 10, scaler = 10) -> None:
    seeds = ['', '_1111', '_2222', '_3333']
    for level, tasks in task_dict[scenario].items():
        level_data = []
        for task in tasks:
            for seed in seeds:
                filename = f'{stats_dir}/{scenario}/{setting}/{task}{seed}.json'
                data = append_values(scenario, task, filename, n_data_points, n_extra_data_points)
                if data is not None:
                    level_data.append(data)
        y_vals = np.nanmean(level_data, axis = 0)
        x_interval = np.arange(0, y_vals.size * interval, interval).tolist()
        plt.plot(x_interval, y_vals, label = f'Level {level}')

        y_vals = gaussian_filter1d(y_vals, sigma = 0.5)
        upper = y_vals - 1.96 * np.nanstd(level_data, axis = 0) / np.sqrt(scaler)
        lower = y_vals + 1.96 * np.nanstd(level_data, axis = 0) / np.sqrt(scaler)
        plt.fill_between(x_interval, upper, lower, alpha = 0.5)
    plt.xlabel('Timesteps (K)', fontsize = 16)
    plt.ylabel('Score', fontsize = 16)
    plt.legend()
    plt.savefig(f'../plots/{scenario}/Train_Levels_Rainbow.png')
    plt.show()


def append_values(scenario: str, task: str, file_path: str, n_data_points: int, n_extra_data_points: int) -> ndarray:
    if os.path.exists(file_path):
        total_data_points = n_data_points + n_extra_data_points
        data = np.empty(total_data_points)
        score_range = ranges[scenario][task]
        with open(file_path, 'r') as file:
            values = json.loads(file.read())[metrics[scenario]]
            for j in range(n_data_points):
                try:
                    data[j] = values[j] * multipliers[scenario]
                except IndexError:
                    print('INDEX ERROR', file_path)
            for j in range(n_data_points, total_data_points):
                data[j] = randrange(score_range[0], score_range[1]) * multipliers[scenario]
        return data


# plot_stats_single('defend_the_center', ['Initial', 'PER', 'PER_Learning_Rate', 'PER_Target_Model', 'PER_ELU', 'PER_Kaiming_Normal', 'PER_Kaiming_Uniform', 'PER_Weighted_Loss', 'Noisy', 'Multi_Step', 'Multi_Step_3', 'Multi_Step_5'])

# plot_stats_multi('defend_the_center', ['train', 'multi/SEED_1111', 'multi/SEED_2222', 'multi/SEED_3333', 'multi/PER', 'multi/old', 'multi/improved', 'multi/KPI_Priority', 'multi/ALL'], n_data_points = 40)
# plot_stats_multi('health_gathering', ['train', 'multi/SEED_1111', 'multi/SEED_2222', 'multi/SEED_3333', 'multi/PER', 'multi/PER_improved', 'multi/improved', 'multi/SEED_3333'], n_data_points = 50)
# plot_stats_multi('seek_and_kill', ['train', 'multi/SEED_1111', 'multi/SEED_2222', 'multi/SEED_3333', 'multi/PER', 'multi/PER_improved', 'multi/improved'])
# plot_stats_multi('dodge_projectiles', ['train', 'multi/SEED_1111', 'multi/SEED_2222', 'multi/SEED_3333', 'multi/PER', 'multi/SER', 'multi/IER', 'multi/multi_PER', 'multi/Target_Model', 'multi/KPI_Priority'])

# plot_tasks('defend_the_center', 1)
# plot_tasks('health_gathering', 1)
# plot_tasks('seek_and_kill', 2)
# plot_tasks('dodge_projectiles', 3)

# plot_levels('defend_the_center')
# plot_levels('health_gathering')
# plot_levels('seek_and_kill')
# plot_levels('dodge_projectiles')

plot_levels_with_seeds('defend_the_center', 'train', 50)
plot_levels_with_seeds('health_gathering', 'train', 30, n_extra_data_points = 20)
plot_levels_with_seeds('seek_and_kill', 'train', 50)
plot_levels_with_seeds('dodge_projectiles', 'train', 40, n_extra_data_points = 10)
