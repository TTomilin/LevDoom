import wandb
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--group-name', type=str, default='dodge_projectiles')

if __name__ == '__main__':
    api = wandb.Api()
    args = parser.parse_args()
    cur_group = args.group_name
    # please change to personal wandb account
    runs = api.runs('xxxx/LevDoom')
    for run in runs:
        if run.group == cur_group:
            file_name = run.name
            algo_name = file_name.split('_')[0]
            seed_num = file_name.split('_')[2]
            save_path = 'logs/{}/{}/seed_{}'.format(cur_group, algo_name, seed_num)
            print(save_path)
            os.makedirs(save_path)
            for file in run.files():
                if 'events' in file.name:
                    file.download(save_path)