import json
import os
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


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
        1: ['lava', 'slime', 'supreme', 'poison', 'obstacles', 'stimpacks', 'shaded_kits', 'water', 'resized_kits', 'resized_agent'],
        2: ['slimy_obstacles', 'shaded_stimpacks', 'supreme_poison'],
        3: ['poison_resized_shaded_kits', 'obstacles_slime_stimpacks', 'lava_supreme_short_agent'],
        4: ['complete']
    },
    'seek_and_kill': {
        0: ['default'],
        1: ['red', 'shadows', 'obstacles', 'invulnerable'],
        2: ['blue_shadows', 'obstacles_resized_enemies', 'red_mixed_enemies'],
        3: ['blue_mixed_resized_enemies', 'red_obstacles_invulnerable', 'resized_shadows_invulnerable'],
        4: ['complete']
    },
    'dodge_projectiles': {
        0: ['default'],
        1: ['barons', 'flames', 'cacodemons', 'resized_agent', 'flaming_skulls', 'city'],
        2: ['revenants', 'city_resized_agent', 'barons_flaming_skulls'],
        3: ['flames_flaming_skulls_mancubus', 'resized_agent_revenants', 'city_arachnotron'],
        4: ['complete']
    }
}


level_dict = {
    'defend_the_center': {
        'default': 0,
        'gore': 1, 'stone_wall': 1, 'fast_enemies': 1, 'mossy_bricks': 1, 'fuzzy_enemies': 1, 'flying_enemies': 1,
        'resized_enemies': 1,
        'gore_mossy_bricks': 2, 'resized_fuzzy_enemies': 2, 'stone_wall_flying_enemies': 2,
        'resized_flying_enemies_mossy_bricks': 3, 'gore_stone_wall_fuzzy_enemies': 3, 'fast_resized_enemies_gore': 3,
        'complete': 4
    },
    'health_gathering': {
        'default': 0,
        'lava': 1, 'slime': 1, 'supreme': 1, 'poison': 1, 'obstacles': 1, 'stimpacks': 1,
        'shaded_kits': 1, 'water': 1, 'resized_kits': 1, 'resized_agent': 1,
        'slimy_obstacles': 2, 'shaded_stimpacks': 2, 'stimpacks_poison': 2, 'resized_kits_lava': 2, 'supreme_poison': 2,
        'poison_resized_shaded_kits': 3, 'obstacles_slime_stimpacks': 3, 'lava_supreme_short_agent': 3,
        'complete': 4
    },
    'seek_and_kill': {
        'default': 0,
        'red': 1, 'blue': 1, 'shadows': 1, 'obstacles': 1, 'invulnerable': 1, 'mixed_enemies': 1, 'resized_enemies': 1,
        'blue_shadows': 2, 'obstacles_resized_enemies': 2, 'red_mixed_enemies': 2,
        'invulnerable_blue': 2, 'resized_enemies_red': 2, 'shadows_obstacles': 2,
        'blue_mixed_resized_enemies': 3, 'red_obstacles_invulnerable': 3, 'resized_shadows_invulnerable': 3,
        'complete': 4
    },
    'dodge_projectiles': {
        'default': 0,
        'barons': 1, 'mancubus': 1, 'flames': 1, 'cacodemons': 1, 'resized_agent': 1, 'flaming_skulls': 1, 'city': 1,
        'revenants': 2, 'arachnotron': 2, 'city_resized_agent': 2, 'barons_flaming_skulls': 2, 'cacodemons_flames': 2,
        'mancubus_resized_agent': 2,
        'flames_flaming_skulls_mancubus': 3, 'resized_agent_revenants': 3, 'city_arachnotron': 3,
        'complete': 4
    }
}


metrics = {
    'defend_the_center': 'frames_alive',
    'health_gathering': 'frames_alive',
    'seek_and_kill': 'kill_count',
    'dodge_projectiles': 'frames_alive'
}


multipliers = {
    'defend_the_center': 4,
    'health_gathering': 4,
    'seek_and_kill': 1,
    'dodge_projectiles': 4
}


protocols = {
    'defend_the_center': 'SEED',
    'health_gathering': 'SEED',
    'seek_and_kill': 'half',
    'dodge_projectiles': 'SEED'
}


def level_data_train(scenario: str, seeds: List[str], levels = [0, 1]):
    level_data = {}
    n_data_points = 20
    multiplier = multipliers[scenario]
    protocol = protocols[scenario]
    for level in levels:
        for task in task_dict[scenario][level]:
            seed_data = np.empty((len(seeds), n_data_points))
            seed_data[:] = np.nan
            for i, seed in enumerate(seeds):
                file_name = f'../statistics/{scenario}/multi/{protocol}{seed}/{task}.json'
                try:
                    with open(file_name, 'r') as f:
                        data = np.array(json.load(f)[metrics[scenario]])
                        for j in range(n_data_points):
                            seed_data[i][j] = data[j] * multiplier
                except FileNotFoundError as error:
                    print(error)
                    continue
                except IndexError as error:
                    print('INDEX ERROR', file_name, error)
            level = level_dict[scenario][task.replace('.json', '')]
            level_data.setdefault(level, []).append(np.array(seed_data))
    return level_data


def level_data_test(scenario: str, method: str, seeds: List[int]):
    log_dir = '../statistics/{}/test/{}'.format(scenario, method)
    tasks = os.listdir(log_dir)
    multiplier = multipliers[scenario]
    level_data = {}
    for task in tasks:
        seed_data = []
        for seed in seeds:
            data = load_seed(log_dir, scenario, task, seed)
            data = [value * multiplier for value in data]
            seed_data.append(data)
        level = level_dict[scenario][task]
        level_data.setdefault(level, []).append(np.array(seed_data))
    return level_data


def load_seed(log_dir: str, scenario: str, task: str, seed: int, num_models = 20):
    protocol = protocols[scenario]
    data_y = []
    for i in range(num_models):
        try:
            with open(f'{log_dir}/{task}/multi_{protocol}_{seed}_v{i}.json', 'r') as f:
                data = json.load(f)
                data_y.append(data[metrics[scenario]][0])
        except FileNotFoundError as error:
            print(error)
            continue
    return data_y


def plot_scenarios(scenarios = list(level_dict.keys()), method = 'rainbow', levels = [0, 1, 2, 3, 4], train_seeds = ['', '_1111', '_2222', '_3333'], test_seeds = [1111, 2222, 3333]):
    plt.rcParams.update({'font.size': 13})
    for scenario in scenarios:
        train_data = level_data_train(scenario, train_seeds)
        test_data = level_data_test(scenario, method, test_seeds)

        level_data = {**train_data, **test_data}

        for level in levels:
            x = np.arange(20) * 10
            data = level_data[level]
            data = [item for sublist in data for item in sublist]
            y = np.nanmean(data, axis = 0)
            y = gaussian_filter1d(y, sigma = 0.5)

            plt.plot(x, y, label = f'Level {level}')
            upper = y - 1.96 * np.nanstd(data, axis = 0) / np.sqrt(30)
            lower = y + 1.96 * np.nanstd(data, axis = 0) / np.sqrt(30)
            plt.fill_between(x, upper, lower, alpha = 0.5)
        plt.title(scenario.replace('_', ' ').title())
        plt.legend()

        plt.xlabel('Timesteps (K)', fontsize = 16)
        plt.ylabel('Score', fontsize = 16)
        plt.savefig(f'../plots/{scenario}/Levels_Rainbow.png')
        plt.show()


if __name__ == '__main__':
    plot_scenarios()
