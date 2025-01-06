import gymnasium as gym

import ale_py

gym.register_envs(ale_py)
# Print available environments
print(gym.envs.registry.keys())

# Load Pong environment
env = gym.make("ALE/Pong-v5", render_mode="human")

# Reset the environment to get the initial state
state, info = env.reset()

# Print initial state information
print("Initial State Shape:", state.shape)

# Take random actions and observe the result
for _ in range(50):
    action = env.action_space.sample()  # Random action
    next_state, reward, done, truncated, info = env.step(action)
    print("Action:", action, "Reward:", reward)

    if done or truncated:
        state, info = env.reset()

env.close()
