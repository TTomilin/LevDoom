import levdoom
from levdoom import Scenario


def main():
    max_steps = 100
    level_envs = levdoom.make_level(Scenario.SEEK_AND_SLAY, level=1, max_steps=max_steps)
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
