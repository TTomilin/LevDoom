import argparse
from util import bool_type


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'GVizDoom runner')

    # Agent arguments
    parser.add_argument(
        '-a', '--algorithm', type = str, default = None, required = True, choices = ['dqn', 'ppo'],
        help = 'DRL algorithm used to construct the model [DQN, DRQN, Dueling_DDQN, DFP, C51] (case-insensitive)'
    )
    parser.add_argument(
        '--frames_per_action', type = int, default = 4,
        help = 'Frame skip count. Number of frames to stack upon each other as input to the model'
    )
    parser.add_argument(
        '--variable_history_size', type = int, default = 5,
        help = 'Number of the most recent game variables kept in the buffer'
    )

    # Training arguments
    parser.add_argument(
        '--device', type = str, default = 'cuda', choices = ['cude', 'cpu'],
        help = 'Device to use for training'
    )
    parser.add_argument(
        '--train', type = bool_type, default = True,
        help = 'Train or evaluate the agent'
    )
    parser.add_argument(
        '-o', '--observe', type = int, default = 10000,
        help = 'Number of iterations to collect experience before training'
    )
    parser.add_argument(
        '-e', '--explore', type = int, default = 50000,
        help = 'Number of iterations to decay the epsilon for'
    )
    parser.add_argument(
        '--learning-rate', type = float, default = 0.0001,
        help = 'Learning rate of the model optimizer'
    )
    parser.add_argument(
        '--max-train-iterations', type = int, default = 10_000_000,
        help = 'Maximum iterations of training'
    )
    parser.add_argument(
        '-g', '--gamma', type = float, default = 0.99,
        help = 'Value of the discount factor for future rewards'
    )
    parser.add_argument(
        '--batch-size', type = int, default = 64,
        help = 'Number of samples in a single training batch'
    )

    # Model arguments
    parser.add_argument(
        '--load-model', type = bool_type, default = False,
        help = 'Load existing weights or train from scratch'
    )
    parser.add_argument(
        '--model-save-frequency', type = int, default = 5000,
        help = 'Number of iterations after which to save a new version of the model'
    )
    parser.add_argument(
        '--model-name-addition', type = str, default = '',
        help = 'An additional identifier to the name of the model. Used to better differentiate stored data'
    )

    # Memory arguments
    parser.add_argument(
        '--load-experience', type = bool_type, default = False,
        help = 'Load existing experience into the replay buffer'
    )
    parser.add_argument(
        '--save-experience', type = bool_type, default = False,
        help = 'Store the gathered experience buffer externally'
    )
    parser.add_argument(
        '--memory-capacity', type = int, default = 50000,
        help = 'Number of transitions to cache in the replay buffer'
    )
    parser.add_argument(
        '--memory-storage-size', type = int, default = 5000,
        help = 'Number of transitions to store externally from the replay buffer'
    )
    parser.add_argument(
        '--memory-update_frequency', type = int, default = 5000,
        help = 'Number of iterations after which to externally store the experiences'
    )

    # Environment arguments
    parser.add_argument(
        '-s', '--scenario', type = str, default = None, required = True,
        choices = ['defend_the_center', 'health_gathering', 'seek_and_kill', 'dodge_projectiles'],
        help = 'Name of the scenario e.g., `defend_the_center` (case-insensitive)'
    )
    parser.add_argument(
        '-t', '--tasks', nargs = "+", default = ['default'],
        help = 'List of tasks, e.g., `default gore stone_wall` (case-insensitive)'
    )
    parser.add_argument(
        '--eval-task', type = str, default = 'default',
        help = 'Task for periodic evaluation, e.g., `fast_enemies` (case-insensitive)'
    )
    parser.add_argument(
        '--n_envs', type = int, default = 1,
        help = 'Number of Doom instances to run in parallel'
    )
    parser.add_argument(
        '--trained-model', type = str, default = None,
        help = 'Name of the trained model which will be used for evaluation, e.g., `mossy_bricks`'
    )
    parser.add_argument(
        '--seed', type = int, default = None,
        help = 'Define a seed to reproducibility'
    )
    parser.add_argument(
        '-m', '--max-epochs', type = int, default = 10000,
        help = 'Maximum number of episodes per scenario task'
    )
    parser.add_argument(
        '-v', '--visualize', type = bool_type, default = False,
        help = 'Visualize the interaction of the agent with the environment'
    )
    parser.add_argument(
        '--sound', type = bool_type, default = False,
        help = 'Enable playing the audio from the game'
    )
    parser.add_argument(
        '--render-hud', type = bool_type, default = True,
        help = 'Render the in-game hud, which displays health, ammo, armour, etc.'
    )
    parser.add_argument(
        '--frame-width', type = int, default = 84,
        help = 'Number of pixels to which the width of the frame is scaled down to'
    )
    parser.add_argument(
        '--frame-height', type = int, default = 84,
        help = 'Number of pixels to which the height of the frame is scaled down to'
    )
    parser.add_argument(
        '--record', type = bool_type, default = False,
        help = 'Record the gameplay'
    )
    parser.add_argument(
        '--game-vars', nargs = "+", default = [], choices = ['position', 'velocity', 'health', 'ammo', 'killcount'],
        help = 'Optional list of game variables to return from the environment at each iteration'
    )
    parser.add_argument(
        '--log-name', type = str, default = 'Train',
        help = 'Name of the experiment'
    )

    # Statistics arguments
    parser.add_argument(
        '--append-statistics', type = bool_type, default = True,
        help = 'Append the rolling statistics to an existing file or overwrite'
    )
    parser.add_argument(
        '--statistics-save-frequency', type = int, default = 5000,
        help = 'Number of iterations after which to write newly aggregated statistics'
    )
    parser.add_argument(
        '--train-report-frequency', type = int, default = 10,
        help = 'Number of iterations after which the training progress is reported'
    )

    # Parse the input arguments
    return parser.parse_args()
