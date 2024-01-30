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
    'defend_the_center': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'resized_flying_enemies_mossy_bricks', 'gore_stone_wall_fuzzy_enemies', \
                             'gore_mossy_bricks', 'fast_resized_enemies_gore', 'complete'],
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
    dqn_acc, rainbow_acc, ppo_acc = 0, 0, 0
    for i in range(len(LOGS[cur_group])):
        print('{} - DQN: {:.2f} +/- {:.2f}'.format(LOGS[cur_group][i], np.mean(np.array(dqn_scores[LOGS[cur_group][i]])[:, -10:]), np.std(np.array(dqn_scores[LOGS[cur_group][i]])[:, -10:])))
        print('{} - Rainbow: {:.2f} +/- {:.2f}'.format(LOGS[cur_group][i], np.mean(np.array(rainbow_scores[LOGS[cur_group][i]])[:, -10:]), np.std(np.array(rainbow_scores[LOGS[cur_group][i]])[:, -10:])))
        print('{} - PPO: {:.2f} +/- {:.2f}'.format(LOGS[cur_group][i], np.mean(np.array(ppo_scores[LOGS[cur_group][i]])[:, -10:]), np.std(np.array(ppo_scores[LOGS[cur_group][i]])[:, -10:])))
        print("=============================================================")
        dqn_acc += np.array(dqn_scores[LOGS[cur_group][i]])[:, -10:]
        rainbow_acc += np.array(rainbow_scores[LOGS[cur_group][i]])[:, -10:]
        ppo_acc += np.array(ppo_scores[LOGS[cur_group][i]])[:, -10:]
    dqn_acc /= len(LOGS[cur_group])
    rainbow_acc /= len(LOGS[cur_group])
    ppo_acc /= len(LOGS[cur_group])
    print('Average - DQN: {:.2f} +/- {:.2f}'.format(np.mean(dqn_acc), np.std(dqn_acc)))
    print('Average - Rainbow: {:.2f} +/- {:.2f}'.format(np.mean(rainbow_acc), np.std(rainbow_acc)))
    print('Average - PPO: {:.2f} +/- {:.2f}'.format(np.mean(ppo_acc), np.std(ppo_acc)))
