import argparse
import pickle

import tensorflow.compat.v1 as tf
from scipy.stats import t

# disable tensorflow-v2
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--group-name', type=str, default='dodge_projectiles')
parser.add_argument('--ylim-top', type=int, default=2100)

ENVIRONMENTS = {
    'defend_the_center': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'gore_mossy_bricks',
                          'gore_stone_wall_fuzzy_enemies', \
                          'resized_flying_enemies_mossy_bricks', 'fast_resized_enemies_gore', 'complete'],
    'health_gathering': ['supreme_poison', 'slime_obstacles', 'shaded_stimpacks', 'poison_resized_shaded_kits', \
                         'obstacles_slime_stimpacks', 'lava_supreme_resized_agent', 'complete'],
    'seek_and_slay': ['blue_shadows', 'obstacles_resized_enemies', 'invulnerable_blue', 'blue_mixed_resized_enemies', \
                      'red_obstacles_invulnerable', 'resized_shadows_red', 'complete'],
    'dodge_projectiles': ['city_resized_agent', 'revenants', 'barons_flaming_skulls', 'city_arachnotron', \
                          'flames_flaming_skulls_mancubus', 'resized_agent_revenants', 'complete'],
}

TRANSLATIONS = {
    # Algorithms
    'dqn': 'DQN',
    'ppo': 'PPO',
    'rainbow': 'Rainbow',

    # Scenarios
    'defend_the_center': 'Defend the Center',
    'health_gathering': 'Health Gathering',
    'seek_and_slay': 'Seek and Slay',
    'dodge_projectiles': 'Dodge Projectiles',

    # Common
    'complete': 'Ultimate Gauntlet',

    # Defend the Center
    'stone_wall_flying_enemies': 'Skyward Stronghold',
    'resized_fuzzy_enemies': 'Giant Phantoms',
    'gore_mossy_bricks': 'Ruined Bastion',
    'gore_stone_wall_fuzzy_enemies': 'Spectral Fortress',
    'resized_flying_enemies_mossy_bricks': 'Colossal Sky Siege',
    'fast_resized_enemies_gore': 'Rapid Rampage',

    # Health Gathering
    'supreme_poison': 'Supreme Poison',
    'slime_obstacles': 'Sludge Barriers',
    'shaded_stimpacks': 'Shady Stimpacks',
    'poison_resized_shaded_kits': 'Toxic Overload',
    'obstacles_slime_stimpacks': 'Mire of Survival',
    'lava_supreme_resized_agent': 'Supreme Lava',

    # Seek and Slay
    'blue_shadows': 'Azure Veil',
    'obstacles_resized_enemies': 'Hazy Maze',
    'invulnerable_blue': 'Cerulean Santuary',
    'blue_mixed_resized_enemies': 'Sapphire Titans',
    'red_obstacles_invulnerable': 'Ruby Ramparts',
    'resized_shadows_red': ' Crimson Eclipse',

    # Dodge Projectiles
    'city_resized_agent': 'Urban Assault',
    'revenants': 'Spectral Hunt',
    'barons_flaming_skulls': 'Infernal Court',
    'city_arachnotron': 'Metro Web',
    'flames_flaming_skulls_mancubus': 'Hellfire Colossus',
    'resized_agent_revenants': 'Colossal Ghosts',
}


def read_pkl(algo: str, log_dir: str, scenario: str) -> tuple:
    with open('{}/{}/{}.pkl'.format(log_dir, scenario, algo), 'rb') as f:
        data = pickle.load(f)
    return data['score'], data['step']


def convert_names(name: str) -> str:
    """
    convert the name to the capital
    """
    str_splits = name.split('_')
    capital_str = ''
    for str_ in str_splits:
        capital_str += '{} '.format(str_)
    return capital_str.title()[:-1]


import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    args = parser.parse_args()
    n_envs = 7
    logdir = 'data'
    algos = ['dqn', 'rainbow', 'ppo']
    scenarios = ENVIRONMENTS.keys()
    seeds = 5
    dof = seeds - 1
    confidence = 0.95
    significance = (1 - confidence) / 2
    plt.style.use('seaborn-deep')
    fig, axarr = plt.subplots(len(scenarios), n_envs + 2, figsize=(16, 9),  # Add two dummy columns for spacing
                              sharey='row', gridspec_kw={
            'width_ratios': [1, 1, 1, 0.01, 1, 1, 1, 0.01, 1]})  # Add extra width for spacing columns

    for j, scenario in enumerate(scenarios):
        envs = ENVIRONMENTS[scenario]
        for i, env in enumerate(envs):
            if i >= 3:  # Adjust index for spacing
                ax = axarr[j, i + 1]
                if i >= 6:  # Adjust index for another spacing
                    ax = axarr[j, i + 2]
            else:
                ax = axarr[j, i]

            bar_locations = np.arange(len(algos))  # Bar locations for each algorithm

            for k, alg in enumerate(algos):
                scores, x_steps = read_pkl(alg, logdir, scenario)
                values = np.array(scores[env])  # Ensure values is a NumPy array
                last_10_avg = np.nanmean(values[:, -10:], axis=1)
                mean = np.nanmean(last_10_avg)
                std = np.nanstd(last_10_avg)
                t_crit = np.abs(t.ppf(significance, dof))
                ci = std * t_crit / np.sqrt(seeds)
                ax.bar(bar_locations[k], mean, width=0.6, yerr=ci, label=TRANSLATIONS[alg], capsize=5, alpha=0.7)

            title_str = TRANSLATIONS[env] if env in TRANSLATIONS else convert_names(env)
            ax.set_xlim([-0.5, len(algos) - 0.5 + 0.25])
            ax.set_xticks(bar_locations)
            # ax.set_xticklabels([TRANSLATIONS[alg] for alg in algos])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_title(title_str, fontsize=8)
            ax.tick_params(labelbottom=True, labelleft=True, labelsize=7)

    # Hide the axes for the dummy spacing columns
    for ax in axarr[:, 3]:
        ax.set_visible(False)
    for ax in axarr[:, 7]:
        ax.set_visible(False)

    # Adding level titles
    fig.text(0.23, 0.98, 'Level 2', ha='center', va='center', fontsize=16)
    fig.text(0.65, 0.98, 'Level 3', ha='center', va='center', fontsize=16)
    fig.text(0.94, 0.98, 'Level 4', ha='center', va='center', fontsize=16)

    handles, labels = fig.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=len(algos), fancybox=True,
               shadow=True, fontsize=12)
    plt.tight_layout(rect=[0.02, 0.05, 1, 0.97])
    # fig.supxlabel('Algorithms', fontsize=15, y=0.07)
    fig.supylabel('Score', fontsize=17)
    plt.savefig('plots/histogram_full.png')
    plt.show()
