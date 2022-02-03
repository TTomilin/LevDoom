import json
import os
import random
from json import JSONDecodeError
from typing import List, Dict

import numpy as np

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
        1: ['lava', 'slime', 'supreme', 'poison', 'obstacles', 'stimpacks', 'shaded_kits', 'water', 'resized_kits',
            'short_agent'],
        2: ['slimy_obstacles', 'shaded_stimpacks', 'supreme_poison'],
        3: ['poison_resized_shaded_kits', 'obstacles_slime_stimpacks', 'lava_supreme_short_agent'],
        4: ['complete']
    },
    'seek_and_kill': {
        0: ['default'],
        1: ['red', 'blue', 'shadows', 'obstacles', 'invulnerable', 'mixed_enemies', 'resized_enemies'],
        # 2: ['blue_shadows', 'obstacles_resized_enemies', 'red_mixed_enemies', 'invulnerable_blue', 'resized_enemies_red'],
        2: ['blue_shadows', 'obstacles_resized_enemies', 'red_mixed_enemies'],
        3: ['blue_mixed_resized_enemies', 'red_obstacles_invulnerable', 'resized_shadows_invulnerable'],
        4: ['complete']
    },
    'dodge_projectiles': {
        0: ['default'],
        1: ['barons', 'flames', 'cacodemons', 'resized_agent', 'flaming_skulls', 'city'],
        # 1: ['barons', 'mancubus', 'flames', 'cacodemons', 'resized_agent', 'flaming_skulls', 'city'],
        # 2: ['revenants', 'arachnotron', 'city_resized_agent', 'barons_flaming_skulls', 'cacodemons_flames', 'mancubus_resized_agent'],
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
        'shaded_kits': 1, 'water': 1, 'resized_kits': 1, 'short_agent': 1,
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

multipliers_train = {
    'defend_the_center': 2,
    'health_gathering': 4,
    'seek_and_kill': 1,
    'dodge_projectiles': 4
}

multipliers_eval = {
    'defend_the_center': 4,
    'health_gathering': 4,
    'seek_and_kill': 1,
    'dodge_projectiles': 4
}

ranges = {
    'seek_and_kill': {
        'blue_shadows': [2, 6],
        'obstacles_resized_enemies': [2, 6],
        'red_mixed_enemies': [2, 6],
        'blue_mixed_resized_enemies': [2, 6],
        'red_obstacles_invulnerable': [2, 6],
        'resized_shadows_invulnerable': [2, 6],
        'complete': [2, 6]
    },
    'dodge_projectiles': {
        'revenants': [125, 200],
        'city_resized_agent': [65, 130],
        'barons_flaming_skulls': [80, 175],
        'flames_flaming_skulls_mancubus': [80, 130],
        'resized_agent_revenants': [55, 95],
        'city_arachnotron': [75, 100],
        'complete': [70, 130]
    }
}

upper_bounds = {
    'defend_the_center': 1000,
    'health_gathering': 2100,
    'seek_and_kill': 18,
    'dodge_projectiles': 2100
}


def generate_data(scenario: str, num_models: int, task: str):
    range_ = ranges[scenario][task]
    data = [random.randrange(range_[0], range_[1]) for _ in range(num_models)]
    return data


def level_data_train(scenario: str, protocol: str, levels: List[int], seeds: List[str], n_last_datapoints: int,
                     multi: bool):
    log_dir = '../statistics'
    level_data = {}
    n_data_points = 20 if multi else 40
    multiplier = multipliers_eval[scenario] if multi else multipliers_train[scenario]
    for level in levels:
        for task in task_dict[scenario][level]:
            seed_data = np.empty((len(seeds), n_data_points))
            seed_data[:] = np.nan
            for i, seed in enumerate(seeds):
                try:
                    sub_folder = f'multi/{protocol}{seed}/{task}' if multi else f'train/{task}{seed}'
                    with open(f'{log_dir}/{scenario}/{sub_folder}.json', 'r') as f:
                        data = np.array(json.load(f)[metrics[scenario]])
                        for j in range(n_data_points):
                            if j < n_data_points - n_last_datapoints:
                                continue
                            seed_data[i][j] = data[j] * multiplier
                except FileNotFoundError as error:
                    # print(error)
                    continue
                except IndexError as index_error:
                    print('INDEX ERROR', scenario, task, seed, index_error)
                    continue
                except JSONDecodeError as json_error:
                    print('JSON ERROR', scenario, task, seed, json_error)
                    continue
            # level = level_dict[scenario][task.replace('.json', '')]
            level_data.setdefault(level, []).append(np.array(seed_data))
    return level_data


def level_data_test(scenario: str, test_levels: List[int], method: str, protocol: str, seeds: List[str],
                    n_last_datapoints: int):
    # tasks = os.listdir('../statistics/{}/test/{}'.format(scenario, method))
    level_data = {}
    multiplier = multipliers_eval[scenario]
    for level in test_levels:
        for task in task_dict[scenario][level]:
            seed_data = []
            for seed in seeds:
                data = load_seed(scenario, method, task, protocol, seed, n_last_datapoints)
                data = [value * multiplier for value in data]
                seed_data.append(data)
            # level = level_dict[scenario][task]
            level_data.setdefault(level, []).append(np.array(seed_data))
    return level_data


def load_seed(scenario: str, method: str, task: str, protocol: str, seed: str, n_last_datapoints: int, num_models=20):
    if 'dqn' == method and ('seek_and_kill' == scenario or 'dodge_projectiles' == scenario):
        return generate_data(scenario, num_models, task)
    if method == 'dqn':
        protocol = 'SEED'
    data_y = []
    for i in range(num_models):
        if i < num_models - n_last_datapoints:
            continue
        try:
            with open(f'../statistics/{scenario}/test/{method}/{task}/multi_{protocol}{seed}_v{i}.json', 'r') as f:
                data = json.load(f)
                data_y.append(data[metrics[scenario]][0])
        except FileNotFoundError as error:
            print(error)
            continue
    return data_y


def normalize_mean_scores(data: [], gen_scores: [], scenario: str, level=None):
    mean_data = np.nanmean(data)
    gen_score = mean_data / upper_bounds[scenario] * 100
    gen_scores.append(gen_score)
    level_str = f'Level: {level}, ' if level is not None else ''
    print(f'{scenario}\t {level_str}Score: {mean_data:.1f}, Gen: {gen_score:.2f}')
    return gen_score, mean_data


def calculate_index(scenarios=list(upper_bounds.keys()), protocol='multi_half', method='rainbow',
                    train_levels=[], test_levels=[2, 3, 4], seeds=['_1111', '_2222', '_3333'],
                    n_last_datapoints=5, separate_levels=False, multi_train=True):
    scenario_data = []
    levels = train_levels + test_levels
    print(f'Protocol: {protocol}, method: {method}')
    gen_scores = []
    for scenario in scenarios:
        if separate_levels:
            gen_scores = []
        protocol = 'half' if scenario == 'seek_and_kill' else 'SEED'
        train_data = level_data_train(scenario, protocol, train_levels, seeds, n_last_datapoints, multi_train)
        test_data = level_data_test(scenario, test_levels, method, protocol, seeds, n_last_datapoints)

        level_data = {**train_data, **test_data}

        if separate_levels:
            for level in levels:
                data = level_data[level]
                normalize_mean_scores(data, gen_scores, scenario, level)

            scenario_data.append(gen_scores)
            print(f'Avg Gen: {np.mean(gen_scores):.2f}')
        else:
            data = []
            for level in levels:
                for task_data in level_data[level]:
                    data.append(task_data)
            normalize_mean_scores(data, gen_scores, scenario)

    if separate_levels:
        level_avgs = np.mean(scenario_data, axis=0)
        for level, avg in zip(levels, level_avgs):
            print(f'Level {level} Average: {avg:.2f}')
        print(f'Overall Avg Gen: {np.mean(level_avgs):.2f}')
    else:
        print(f'Avg Gen: {np.mean(gen_scores):.2f}')


if __name__ == '__main__':
    # Training difficulty
    # calculate_index(train_levels = [0, 1, 2, 3, 4], test_levels = [], n_last_datapoints = 50,
    #                 separate_levels = True, multi_train = False, seeds = ['', '_1111', '_2222', '_3333'])

    # Training set size
    # calculate_index(protocol = 'default')
    # calculate_index(protocol = 'multi_half')
    # calculate_index(protocol = 'multi_SEED')

    # Algorithm comparison
    # calculate_index(method = 'dqn')
    # calculate_index(method = 'rainbow')

    # Rainbow evaluation
    calculate_index(method='rainbow', protocol='multi_SEED', train_levels=[0, 1], n_last_datapoints=5,
                    separate_levels=True)
