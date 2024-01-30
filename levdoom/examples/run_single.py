import levdoom


def main():
    env = levdoom.make('HealthGatheringLevel1_1-v0')
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space)

    env.reset()
    done = False
    steps = 0
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        env.render()
        steps += 1
        total_reward += reward
    print(f"Episode finished in {steps} steps. Reward: {total_reward:.2f}")
    env.close()


if __name__ == '__main__':
    main()
