import tensorflow.compat.v1 as tf
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
import argparse
# disable tensorflow-v2
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--group-name', type=str, default='dodge_projectiles')

# logs name
LOGS = {
    'dodge_projectiles': ['city_resized_agent', 'revenants', 'barons_flaming_skulls', 'city_arachnotron', \
                            'flames_flaming_skulls_mancubus', 'resized_agent_revenants', 'complete'], 
    'health_gathering': ['default', 'lava', 'slime', 'supreme', 'poison', 'obstacles', 'stimpacks', 'shaded_kits', 'resized_kits', 'shaded_stimpacks', 'supreme_poison', 'slime_obstacles', 'poison_resized_shaded_kits', \
                         'obstacles_slime_stimpacks', 'lava_supreme_resized_agent', 'complete'],
    'defend_the_center': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'resized_flying_enemies_mossy_bricks', 'gore_stone_wall_fuzzy_enemies', \
                             'gore_mossy_bricks', 'fast_resized_enemies_gore', 'complete'],
}

def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def extract_info(algo, logdir, cur_group):
    print('Read data from algorithm: {}'.format(algo))
    scores = {}
    for key in LOGS[cur_group]:
        scores[key] = []
    for idx in range(5):
        cur_path = '{}/{}/{}/seed_{}'.format(logdir, cur_group, algo, idx+1)
        event_path = glob("{}/events*".format(cur_path))[0]
        for key in LOGS[cur_group]:
            target_tag = ['test_{}/length'.format(key), 'train_{}/length'.format(key)]
            score_ = []
            for e in tf.train.summary_iterator(event_path):
                for v in e.summary.value:
                    if v.tag in target_tag:
                        #step_.append(e.step)
                        score_.append(v.simple_value)
            steps = np.arange(100) * 100000
            if len(score_) != 100:
                score_ = score_[::int(len(score_) / 100)][:100]
            steps, score_ = smooth_reward_curve(steps, score_)
            scores[key].append(score_)
    return scores, steps  

def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    cur_group = args.group_name
    logdir = '../logs'
    algos = ['dqn', 'rainbow', 'ppo']
    os.makedirs('data/{}'.format(cur_group), exist_ok=True)
    # extract the information
    rainbow_scores, x_steps = extract_info('rainbow', logdir, cur_group)
    # write data
    save_pickle('data/{}/rainbow_train.pkl'.format(cur_group), {'score': rainbow_scores, 'step': x_steps})