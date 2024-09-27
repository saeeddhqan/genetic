import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human", continuous=True)
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   print(action)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset(seed=42)
      print('x'*15)
env.close()
