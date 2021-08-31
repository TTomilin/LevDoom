import argparse
import os

from vizdoom import Mode, ScreenResolution, DoomGame

parser = argparse.ArgumentParser(description = 'GViZDoom runner')

parser.add_argument(
    '-s', '--scenario', type = str, default = None, required = True,
    help = 'Name of the scenario e.g., `defend_the_center` (case-insensitive)'
)
parser.add_argument(
    '-t', '--task', type = str, default = None, required = True,
    help = 'Name of the task, e.g., `default gore stone_wall` (case-insensitive)'
)
parser.add_argument(
    '-e', '--epochs', type = int, default = 3,
    help = 'Number of recorded episodes'
)

# Parse the input arguments
args = parser.parse_args()
print('Running GViZDoom with the following configuration')
for key, val in args.__dict__.items():
    print(f'{key}: {val}')

# Find the root directory
root_dir = os.path.dirname(os.getcwd())
if 'GVizDoom' not in root_dir:
    root_dir += '/GVizDoom'  # Fix for external script executor

game = DoomGame()
game.load_config(f'{root_dir}/scenarios/{args.scenario}/{args.scenario}.cfg')
game.set_doom_scenario_path(f'{root_dir}/scenarios/{args.scenario}/{args.task}.wad')
game.set_sound_enabled(True)
game.set_window_visible(True)
game.set_render_hud(True)
game.set_mode(Mode.SPECTATOR)
game.set_screen_resolution(ScreenResolution.RES_800X600)
game.init()

for i in range(args.epochs):
    game.replay_episode(f'recordings/{args.scenario}/{args.task}/episode{i}.lmp')

    while not game.is_episode_finished():
        # Use advance_action instead of make_action.
        game.advance_action()
