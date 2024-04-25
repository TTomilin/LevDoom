import argparse
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
from scipy.stats import t

# disable tensorflow-v2
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--group-name', type=str, default='seek_and_slay')

# logs name
LOGS = {
    'dodge_projectiles': ['city_resized_agent', 'revenants', 'barons_flaming_skulls', 'city_arachnotron', \
                          'flames_flaming_skulls_mancubus', 'resized_agent_revenants', 'complete'],
    'health_gathering': ['supreme_poison', 'slime_obstacles', 'shaded_stimpacks', 'poison_resized_shaded_kits', \
                         'obstacles_slime_stimpacks', 'lava_supreme_resized_agent', 'complete'],
    'defend_the_center': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'resized_flying_enemies_mossy_bricks',
                          'gore_stone_wall_fuzzy_enemies', \
                          'gore_mossy_bricks', 'fast_resized_enemies_gore', 'complete'],
}

LOGS_Level = {
    'defend_the_center': {'Level 2': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'gore_mossy_bricks'], \
                          'Level 3': ['resized_flying_enemies_mossy_bricks', 'gore_stone_wall_fuzzy_enemies',
                                      'fast_resized_enemies_gore'], \
                          'Level 4': ['complete']
                          },
    'health_gathering': {'Level 2': ['supreme_poison', 'slime_obstacles', 'shaded_stimpacks'], \
                         'Level 3': ['poison_resized_shaded_kits', 'obstacles_slime_stimpacks',
                                     'lava_supreme_resized_agent'], \
                         'Level 4': ['complete']
                         },
    'dodge_projectiles': {'Level 2': ['revenants', 'city_resized_agent', 'barons_flaming_skulls'], \
                          'Level 3': ['city_arachnotron', 'flames_flaming_skulls_mancubus', 'resized_agent_revenants'], \
                          'Level 4': ['complete']
                          },
    'seek_and_slay': {'Level 2': ['blue_shadows', 'obstacles_resized_enemies', 'invulnerable_blue'], \
                      'Level 3': ['blue_mixed_resized_enemies', 'red_obstacles_invulnerable', 'resized_shadows_red'], \
                      'Level 4': ['complete']
                      },
}


def read_pkl(algo, logdir, cur_group):
    with open('{}/{}/{}.pkl'.format(logdir, cur_group, algo), 'rb') as f:
        data = pickle.load(f)
    return data['score'], data['step']


def convert_names(str):
    """
    convert the name to the capital
    """
    str_splits = str.split('_')
    capital_str = ''
    for str_ in str_splits:
        capital_str += '{} '.format(str_)
    return capital_str.title()[:-1]


if __name__ == '__main__':
    plt.style.use('seaborn-deep')
    args = parser.parse_args()
    font_size = 30
    cur_group = args.group_name
    logdir = 'data'
    algos = ['dqn', 'rainbow', 'ppo']
    confidence = 0.95
    seeds = 5
    dof = seeds - 1
    significance = (1 - confidence) / 2
    #
    rainbow_scores, x_steps = read_pkl('rainbow', logdir, cur_group)
    x_steps = x_steps / (10**7)
    # plot
    plt.figure(figsize=(12, 8))
    _, ax = plt.subplots(figsize=(12, 8))
    # extract information from different level
    for level in LOGS_Level[cur_group].keys():
        # extract info
        total_rewards = 0
        for task_name in LOGS_Level[cur_group][level]:
            total_rewards += np.array(rainbow_scores[task_name])
        total_rewards /= len(LOGS_Level[cur_group][level])

        # plots
        mean = np.nanmean(total_rewards, axis=0)
        std = np.nanstd(total_rewards, axis=0)

        t_crit = np.abs(t.ppf(significance, dof))
        ci = std * t_crit / np.sqrt(seeds)

        plt.plot(x_steps, mean, label=level, linewidth=3)
        plt.fill_between(x_steps, mean - ci, mean + ci, alpha=0.3)

        title_str = convert_names(cur_group)
        plt.xlim([0, 1])
        plt.title(title_str, fontsize=font_size)
        plt.xlabel('Timesteps ' + r'$(\times 10^{7})$', fontsize=font_size - 5)
        plt.ylabel('Score', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        # plt.legend(loc='lower right', prop={'size': font_size})
    if cur_group == 'dodge_projectiles':
        plt.legend(loc='lower right', prop={'size': font_size})
    plt.tight_layout()
    plt.savefig('plots/curve_{}_level.png'.format(cur_group))
    plt.show()
