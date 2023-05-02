import argparse
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
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
    'dqn': 'DQN',
    'ppo': 'PPO',
    'rainbow': 'Rainbow',

    'defend_the_center': 'Defend the Center',
    'health_gathering': 'Health Gathering',
    'seek_and_slay': 'Seek and Slay',
    'dodge_projectiles': 'Dodge Projectiles',
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
    fig, ax = plt.subplots(len(scenarios), n_envs, sharey='row', sharex='all', figsize=(16, 9))
    for j, scenario in enumerate(scenarios):
        envs = ENVIRONMENTS[scenario]
        for i, env in enumerate(envs):
            for k, alg in enumerate(algos):
                scores, x_steps = read_pkl(alg, logdir, scenario)
                x_steps = x_steps / 10**7
                values = scores[env]
                mean = np.nanmean(values, axis=0)
                std = np.nanstd(values, axis=0)

                t_crit = np.abs(t.ppf(significance, dof))
                ci = std * t_crit / np.sqrt(seeds)

                ax[j][i].plot(x_steps, mean, label=TRANSLATIONS[alg], linewidth=2)
                ax[j, i].fill_between(x_steps, mean - ci, mean + ci, alpha=0.3)

            title_str = convert_names(env)
            ax[j, i].set_xlim([0, 1])
            ax[j, i].set_title(title_str, fontsize=8)
            ax[j, i].tick_params(labelbottom=True, labelleft=True, labelsize=7)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=len(algos), fancybox=True,
               shadow=True, fontsize=12)
    plt.tight_layout(rect=[0.02, 0.1, 1, 1])
    fig.supxlabel('Timesteps ' + r'$(\times 10^{7})$', fontsize=15, y=0.07)
    fig.supylabel('Score', fontsize=17)
    plt.savefig('curve_main.png')
    plt.show()
