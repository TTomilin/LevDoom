import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='data')
parser.add_argument('--output', type=str, default=None, help='Write LaTeX to file instead of stdout')

SCENARIOS = {
    'defend_the_center': 'Defend the Center',
    'health_gathering': 'Health Gathering',
    'seek_and_slay': 'Seek and Slay',
    'dodge_projectiles': 'Dodge Projectiles',
}

ACRONYMS = {
    'defend_the_center': 'DtC',
    'health_gathering': 'HG',
    'seek_and_slay': 'SaS',
    'dodge_projectiles': 'DP',
}

# Environments available in the pkl files, grouped by level.
# First 3 entries belong to Level 2, next 3 to Level 3, last 1 to Level 4.
ENVIRONMENTS = {
    'defend_the_center': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'gore_mossy_bricks',
                          'resized_flying_enemies_mossy_bricks', 'gore_stone_wall_fuzzy_enemies', 'fast_resized_enemies_gore',
                          'complete'],
    'health_gathering': ['supreme_poison', 'slime_obstacles', 'shaded_stimpacks',
                         'poison_resized_shaded_kits', 'obstacles_slime_stimpacks', 'lava_supreme_resized_agent',
                         'complete'],
    'seek_and_slay': ['blue_shadows', 'obstacles_resized_enemies', 'invulnerable_blue',
                      'blue_mixed_resized_enemies', 'red_obstacles_invulnerable', 'resized_shadows_red',
                      'complete'],
    'dodge_projectiles': ['city_resized_agent', 'revenants', 'barons_flaming_skulls',
                          'city_arachnotron', 'flames_flaming_skulls_mancubus', 'resized_agent_revenants',
                          'complete'],
}

ENVIRONMENTS_BY_LEVEL = {
    scenario: {
        2: envs[:3],
        3: envs[3:6],
        4: envs[6:],
    }
    for scenario, envs in ENVIRONMENTS.items()
}

LEVELS = [2, 3, 4]
ALGOS = ['dqn', 'rainbow', 'ppo']
ALGO_NAMES = {'dqn': 'DQN', 'rainbow': 'Rainbow', 'ppo': 'PPO'}


def read_pkl(algo, logdir, scenario):
    with open(f'{logdir}/{scenario}/{algo}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['score']


def level_stats(scores_dict, env_list):
    """Average raw seed×epoch arrays across environments, then return mean ± std."""
    acc = None
    for env in env_list:
        arr = np.array(scores_dict[env])[:, -10:]  # (n_seeds, 10)
        acc = arr if acc is None else acc + arr
    acc = acc / len(env_list)
    return np.mean(acc), np.std(acc)


def fmt_cell(mean, std, bold):
    s = f'{mean:.1f}$\\pm${std:.1f}'
    return f'\\textbf{{{s}}}' if bold else s


def generate_table(logdir):
    # Compute stats: all_stats[scenario][algo] = {level: (mean, std), 'avg': (mean, std)}
    all_stats = {}
    for scenario in SCENARIOS:
        all_stats[scenario] = {}
        for algo in ALGOS:
            scores = read_pkl(algo, logdir, scenario)
            all_stats[scenario][algo] = {}
            for level in LEVELS:
                envs = ENVIRONMENTS_BY_LEVEL[scenario][level]
                all_stats[scenario][algo][level] = level_stats(scores, envs)
            # Overall average: equal weight per environment across all levels
            all_envs = ENVIRONMENTS[scenario]
            all_stats[scenario][algo]['avg'] = level_stats(scores, all_envs)

    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\caption{Quantitative comparison between DQN, Rainbow and PPO across four scenarios:'
                 r' Defend the Center (DtC), Health Gathering (HG), Seek and Slay (SaS), and Dodge Projectiles (DP).'
                 r' We train the agents on all environments of levels 0 and 1, and evaluate them on environments'
                 r' of higher levels. Results are shown as mean $\pm$ standard deviation over the last 10 evaluation'
                 r' epochs across five seeds, averaged over all environments within each level.'
                 r' The highest scores are in \textbf{bold}.}')
    lines.append(r'    \centering')
    lines.append(r'    \resizebox{\textwidth}{!}{')
    lines.append(r'    \newcolumntype{C}{>{\centering\arraybackslash}m{2.5cm}}')
    lines.append(r'    \begin{tabular}{@{}lrCCC@{\quad}C@{}}')
    lines.append(r'    \toprule')
    lines.append(r'    Scenario & Algorithm & Level 2 & Level 3 & Level 4 & Average\\')
    lines.append(r'    \midrule')

    for s_idx, (scenario, scenario_name) in enumerate(SCENARIOS.items()):
        acronym = ACRONYMS[scenario]

        # Find best algo per column for bold
        col_keys = LEVELS + ['avg']
        best = {}
        for col in col_keys:
            best[col] = max(ALGOS, key=lambda a: all_stats[scenario][a][col][0])

        for a_idx, algo in enumerate(ALGOS):
            cells = []
            for col in col_keys:
                mean, std = all_stats[scenario][algo][col]
                cells.append(fmt_cell(mean, std, bold=(best[col] == algo)))
            row = ' & '.join([ALGO_NAMES[algo]] + cells)
            if a_idx == 0:
                multirow = f'    \\multirow{{{len(ALGOS)}}}{{*}}{{{acronym}}}'
                lines.append(f'{multirow} & {row} \\\\')
            else:
                lines.append(f'    & {row} \\\\')

        if s_idx < len(SCENARIOS) - 1:
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'    \end{tabular}')
    lines.append(r'    }')
    lines.append(r'    \label{tab:quantitative_comp_levels}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


if __name__ == '__main__':
    args = parser.parse_args()
    table = generate_table(args.logdir)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(table)
        print(f'Written to {args.output}')
    else:
        print(table)
