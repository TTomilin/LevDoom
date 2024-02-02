import levdoom
from levdoom import Scenario


def main():
    max_steps = 50
    for name, scenario in Scenario.__members__.items():
        print("Running scenario", name.lower())
        for level in range(5):
            print("Running level", level)
            level_envs = levdoom.make_level(scenario, level=level, max_steps=max_steps)
            for env in level_envs:
                env.reset()
                total_reward = 0
                for i in range(max_steps):
                    action = env.action_space.sample()
                    state, reward, done, truncated, info = env.step(action)
                    env.render()
                    total_reward += reward
                    if done or truncated:
                        break
                print(f"{env.unwrapped.name} finished in {i + 1} steps. Reward: {total_reward:.2f}")
                env.close()


if __name__ == '__main__':
    main()
