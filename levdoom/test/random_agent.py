import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from levdoom.utils.enums import DoomScenario


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--scenario', type=str, default='defend_the_center', help='Scenario to use')
    parser.add_argument('--task', type=str, default='default', help='Task to use')
    parser.add_argument('--max_episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--render_sleep', type=float, default=0.05, help='Time to sleep between rendering')
    parser.add_argument('--render', type=bool, default=True, help='Whether to render the environment')
    parser.add_argument('--watch', type=bool, default=True, help='Whether to watch the environment')
    parser.add_argument('--frame_height', type=int, default=84, help='Height of the rendered frame')
    parser.add_argument('--frame_width', type=int, default=84, help='Width of the rendered frame')
    parser.add_argument('--frame_stack', type=int, default=4, help='Number of frames to stack')
    parser.add_argument('--frame_skip', type=int, default=4, help='Number of frames to skip')
    parser.add_argument('--variable_queue_len', type=int, default=5, help='Length of the variable queue')
    return parser.parse_args()


def run(args: Namespace) -> None:
    args.experiment_dir = Path(__file__).parent.parent.resolve()
    print('Experiment directory', args.experiment_dir)

    # Determine scenario class
    if args.scenario.upper() not in DoomScenario._member_names_:
        raise ValueError(f'Unknown scenario provided: `{args.scenario}`')
    scenario_class = DoomScenario[args.scenario.upper()].value

    args.cfg_path = f"{args.experiment_dir}/maps/{args.scenario}/{args.scenario}.cfg"
    args.res = (args.frame_skip, args.frame_height, args.frame_width)
    env = scenario_class(args, args.task)
    args.state_shape = args.res
    args.action_shape = env.action_space.shape or env.action_space.n
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space)

    for ep in range(args.max_episodes):
        env.reset()
        done = False
        steps = 0
        rewards = []
        while not done:
            state, reward, done, info = env.step(env.action_space.sample())
            steps += 1
            rewards.append(reward)
            if args.render:
                env.render()
                time.sleep(args.render_sleep)
        print(f"Episode {ep + 1} reward: {sum(rewards)}, steps: {steps}")
    env.close()


if __name__ == '__main__':
    run(get_args())
