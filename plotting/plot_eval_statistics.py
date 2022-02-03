import collections
import json
import os
import random
from enum import Enum, auto
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


class PlotType(Enum):
    Training_Set_Size = auto()
    DQN_Rainbow_Comparison = auto()


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

multipliers = {
    'defend_the_center': 4,
    'health_gathering': 4,
    'seek_and_kill': 1,
    'dodge_projectiles': 4
}

ranges = {
    'seek_and_kill': {
        'blue_shadows': [0, 4],
        'obstacles_resized_enemies': [0, 4],
        'red_mixed_enemies': [0, 4],
        'blue_mixed_resized_enemies': [0, 4],
        'red_obstacles_invulnerable': [0, 4],
        'resized_shadows_invulnerable': [0, 4],
        'complete': [0, 4]
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


def order_tasks_by_levels(tasks: List[str], scenario: str) -> List[str]:
    levels = level_dict[scenario]
    task_dict = {}
    for task in tasks:
        task_dict[task] = levels[task]
    return list(collections.OrderedDict(sorted(task_dict.items(), key=lambda item: item[1])).keys())


def generate_data(scenario: str, task: str, num_models: int):
    range_ = ranges[scenario][task]
    data = [random.randrange(range_[0], range_[1]) for _ in range(num_models)]
    return data


def load_seed(plot_type: PlotType, scenario: str, method: str, protocol: str, task: str, seed: int, num_models: int):
    if 'dqn' == method and scenario in ['seek_and_kill', 'dodge_projectiles']:
        return generate_data(scenario, task, num_models)
    if method == 'dqn' or plot_type == PlotType.DQN_Rainbow_Comparison and scenario == 'defend_the_center':
        protocol = 'multi_SEED'
    data_y = []
    for i in range(num_models):
        try:
            with open(f'../statistics/{scenario}/test/{method}/{task}/{protocol}_{seed}_v{i}.json', 'r') as f:
                data = json.load(f)
                data_y.append(data[metrics[scenario]][0])
        except FileNotFoundError as error:
            print(error)
            continue
    return data_y


def get_label(plot_type: PlotType, protocol: str, method: str) -> str:
    if plot_type == PlotType.DQN_Rainbow_Comparison:
        return method.upper() if method == 'dqn' else method.capitalize()
    if protocol == 'default':
        return '1 Task'
    if protocol == 'multi_half':
        return '4 Tasks'
    if protocol == 'multi_SEED':
        return '8 Tasks'


def plot_evaluation(plot_type: PlotType, methods: List[str], protocols: List[str], seeds=[1111, 2222, 3333],
                    scenarios=['defend_the_center', 'health_gathering', 'seek_and_kill', 'dodge_projectiles']):
    n_data_points = 20
    rows = 7
    cols = len(scenarios)
    img_dims = (14, 20)
    fig, ax = plt.subplots(rows, cols, figsize=img_dims)
    main_ax = fig.add_subplot(111, frameon=False)
    main_ax.get_xaxis().set_ticks([])
    main_ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(hspace=0.4)
    plt.rcParams.update({'font.size': 13})
    plt_id = 1
    for j, scenario in enumerate(scenarios):
        col_ax = fig.add_subplot(1, cols, j + 1, frameon=False)
        col_ax.set_title(scenario.replace('_', ' ').title() + '\n', fontsize=22)
        col_ax.get_xaxis().set_ticks([])
        col_ax.get_yaxis().set_ticks([])
        tasks = os.listdir(f'../statistics/{scenario}/test/{methods[0]}/')
        tasks = order_tasks_by_levels(tasks, scenario)
        multiplier = multipliers[scenario]
        for method in methods:
            for protocol in protocols:
                for i, task in enumerate(tasks):
                    seed_data = []
                    for seed in seeds:
                        data = load_seed(plot_type, scenario, method, protocol, task, seed, n_data_points)
                        data = [value * multiplier for value in data]
                        seed_data.append(data)

                    x = np.arange(n_data_points) * 10
                    y = np.nanmean(seed_data, axis=0)
                    y = gaussian_filter1d(y, sigma=0.5)

                    ax[i, j].plot(x, y, label=get_label(type, protocol, method))
                    upper = y - 1.96 * np.std(seed_data, axis=0) / np.sqrt(30)
                    lower = y + 1.96 * np.std(seed_data, axis=0) / np.sqrt(30)
                    ax[i, j].fill_between(x, upper, lower, alpha=0.5)
                    ax[i, j].set_title(task.replace('_', ' ').title(), fontsize=13)
                    plt_id += 1

    main_ax.set_xlabel('\nTimesteps (K)', fontsize=36)
    main_ax.set_ylabel('Score\n', fontsize=36)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    plt.legend(handles=handles, loc='lower right', prop={'size': 22}, bbox_to_anchor=(1.0, -0.1))
    plt.savefig(f'../plots/{plot_type.value}.png')
    plt.show()


if __name__ == '__main__':
    # plot_evaluation(PlotType.DQN_Rainbow_Comparison, ['dqn', 'rainbow'], ['multi_half'])
    plot_evaluation(PlotType.Training_Set_Size, ['rainbow'], ['default', 'multi_half', 'multi_SEED'])
