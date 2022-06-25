import tensorflow.compat.v1 as tf
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import pickle
import argparse
# disable tensorflow-v2
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--group-name', type=str, default='dodge_projectiles')

# logs name
LOGS = {
    'dodge_projectiles': ['city_resized_agent', 'revenants', 'barons_flaming_skulls', 'city_arachnotron', \
                            'flames_flaming_skulls_mancubus', 'resized_agent_revenants', 'complete'],
    'health_gathering': ['supreme_poison', 'slime_obstacles', 'shaded_stimpacks', 'poison_resized_shaded_kits', \
                         'obstacles_slime_stimpacks', 'lava_supreme_resized_agent', 'complete'],
    'defend_the_center': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'gore_mossy_bricks', 'gore_stone_wall_fuzzy_enemies', \
                             'resized_flying_enemies_mossy_bricks', 'fast_resized_enemies_gore', 'complete'],
    'seek_and_slay': ['blue_shadows', 'obstacles_resized_enemies', 'invulnerable_blue', 'blue_mixed_resized_enemies', \
                             'red_obstacles_invulnerable', 'resized_shadows_red', 'complete'],
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
    args = parser.parse_args()
    font_size = 30
    cur_group = args.group_name
    logdir = 'data'
    algos = ['dqn', 'rainbow', 'ppo']
    #
    dqn_scores, x_steps = read_pkl('dqn', logdir, cur_group)
    x_steps = x_steps / (10**7)
    rainbow_scores, _ = read_pkl('rainbow', logdir, cur_group)
    ppo_scores, _ = read_pkl('ppo', logdir, cur_group)
    # plot
    plt.figure(figsize=(12, 60))
    _, ax = plt.subplots(figsize=(12, 60))
    for i in range(len(LOGS[cur_group])):
        plt.subplot(7,1,i+1)
        plt.plot(x_steps, np.nanmedian(dqn_scores[LOGS[cur_group][i]], axis=0), label='DQN')
        plt.fill_between(x_steps, np.nanpercentile(dqn_scores[LOGS[cur_group][i]], 25, axis=0), np.nanpercentile(dqn_scores[LOGS[cur_group][i]], 75, axis=0), alpha=0.25)
        plt.plot(x_steps, np.nanmedian(rainbow_scores[LOGS[cur_group][i]], axis=0), label='Rainbow')
        plt.fill_between(x_steps, np.nanpercentile(rainbow_scores[LOGS[cur_group][i]], 25, axis=0), np.nanpercentile(rainbow_scores[LOGS[cur_group][i]], 75, axis=0), alpha=0.25)
        plt.plot(x_steps, np.nanmedian(ppo_scores[LOGS[cur_group][i]], axis=0), label='PPO')
        plt.fill_between(x_steps, np.nanpercentile(ppo_scores[LOGS[cur_group][i]], 25, axis=0), np.nanpercentile(ppo_scores[LOGS[cur_group][i]], 75, axis=0), alpha=0.25)
        title_str = convert_names(LOGS[cur_group][i])
        plt.xlim([0, 1])
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.title(title_str, fontsize=font_size)
        plt.xlabel('Timesteps ' + r'$(\times 10^{7})$', fontsize=font_size-5)
        #ax.xaxis.get_offset_text().set_fontsize(font_size)
        plt.ylabel('Score', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
    if cur_group == 'dodge_projectiles':
        plt.legend(loc='lower right', prop={'size': font_size})
    plt.tight_layout()
    plt.savefig('curve_{}.pdf'.format(cur_group))