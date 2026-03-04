import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='data')

ENVIRONMENTS = {
    'defend_the_center': ['stone_wall_flying_enemies', 'resized_fuzzy_enemies', 'gore_mossy_bricks',
                          'gore_stone_wall_fuzzy_enemies',
                          'resized_flying_enemies_mossy_bricks', 'fast_resized_enemies_gore', 'complete'],
    'health_gathering': ['supreme_poison', 'slime_obstacles', 'shaded_stimpacks', 'poison_resized_shaded_kits',
                         'obstacles_slime_stimpacks', 'lava_supreme_resized_agent', 'complete'],
    'seek_and_slay': ['blue_shadows', 'obstacles_resized_enemies', 'invulnerable_blue', 'blue_mixed_resized_enemies',
                      'red_obstacles_invulnerable', 'resized_shadows_red', 'complete'],
    'dodge_projectiles': ['city_resized_agent', 'revenants', 'barons_flaming_skulls', 'city_arachnotron',
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
    'complete': 'Final Gauntlet',

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
    'resized_shadows_red': 'Crimson Eclipse',

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
    return ' '.join(part.capitalize() for part in name.split('_'))


if __name__ == '__main__':
    args = parser.parse_args()
    n_envs = 7  # 3 Level 2 + 3 Level 3 + 1 Level 4
    logdir = args.logdir
    algos = ['dqn', 'rainbow', 'ppo']
    scenarios = list(ENVIRONMENTS.keys())
    seeds = 5
    dof = seeds - 1
    confidence = 0.95
    significance = (1 - confidence) / 2

    tick_fontsize = 12
    legend_fontsize = 14
    subtitle_fontsize = 11
    scenario_header_fontsize = 13
    level_label_fontsize = 14

    plt.style.use('seaborn-v0_8-deep')

    # Grid: rows = environments (grouped by level), cols = scenarios.
    # Two dummy rows (height ~0) act as visual gaps between level groups.
    n_grid_rows = n_envs + 2  # 9 rows total
    n_grid_cols = len(scenarios)  # 4 columns

    fig, axarr = plt.subplots(
        n_grid_rows, n_grid_cols,
        figsize=(8, 10),
        sharey='col',
        gridspec_kw={'height_ratios': [1, 1, 1, 0.0, 1, 1, 1, 0.0, 1]},
    )

    for j, scenario in enumerate(scenarios):
        envs = ENVIRONMENTS[scenario]
        for i, env in enumerate(envs):
            # Map env index to actual grid row, skipping the two spacing rows at 3 and 7.
            if i >= 6:
                row = i + 2
            elif i >= 3:
                row = i + 1
            else:
                row = i

            ax = axarr[row, j]
            bar_locations = np.arange(len(algos))

            for k, alg in enumerate(algos):
                scores, _ = read_pkl(alg, logdir, scenario)
                values = np.array(scores[env])
                last_10_avg = np.nanmean(values[:, -10:], axis=1)
                mean = np.nanmean(last_10_avg)
                std = np.nanstd(last_10_avg)
                t_crit = np.abs(t.ppf(significance, dof))
                ci = std * t_crit / np.sqrt(seeds)
                ax.bar(bar_locations[k], mean, width=0.8, yerr=ci,
                       label=TRANSLATIONS[alg], capsize=5, alpha=0.7)

            title_str = TRANSLATIONS[env] if env in TRANSLATIONS else convert_names(env)
            ax.set_xlim([-0.5, len(algos) - 0.5 + 0.25])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_title(title_str, fontsize=subtitle_fontsize)
            ax.tick_params(labelbottom=True, labelleft=True, labelsize=tick_fontsize)

    # Hide the dummy spacing rows.
    for ax in axarr[3, :]:
        ax.set_visible(False)
    for ax in axarr[7, :]:
        ax.set_visible(False)

    handles, labels = fig.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(algos), fancybox=True, shadow=True, fontsize=legend_fontsize)

    # Leave space at top for scenario headers and level labels.
    plt.tight_layout(rect=[0.04, 0.04, 1.0, 0.9])

    # Horizontal center of the full plot area (spans all scenario columns).
    plot_x0 = axarr[0, 0].get_position().x0
    plot_x1 = axarr[0, n_grid_cols - 1].get_position().x1
    plot_center_x = (plot_x0 + plot_x1) / 2

    # Level labels placed horizontally at the top of each level band.
    # Level 2: just above the top row (row 0), below scenario names.
    # Level 3/4: in the centre of the respective spacing rows (rows 3 and 7).
    level2_y = axarr[0, 0].get_position().y1 + 0.03
    level3_y = (axarr[3, 0].get_position().y0 + axarr[3, 0].get_position().y1) / 2
    level4_y = (axarr[7, 0].get_position().y0 + axarr[7, 0].get_position().y1) / 2

    for y, label in [(level2_y, 'Level 2'), (level3_y, 'Level 3'), (level4_y, 'Level 4')]:
        fig.text(plot_center_x, y, label, ha='center', va='center', fontsize=level_label_fontsize)

    # Scenario names above each column with extra clearance above the level 2 label.
    for j, scenario in enumerate(scenarios):
        x = (axarr[0, j].get_position().x0 + axarr[0, j].get_position().x1) / 2
        y = axarr[0, j].get_position().y1 + 0.05
        fig.text(x, y, TRANSLATIONS[scenario], ha='center', va='bottom',
                 fontsize=scenario_header_fontsize, fontweight='bold')

    fig.supylabel('Score', fontsize=17)
    plt.savefig('plots/histogram_main_results.png', bbox_inches='tight')
    plt.savefig('plots/histogram_main_results.pdf', bbox_inches='tight')
    plt.show()
