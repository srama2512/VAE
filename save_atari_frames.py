import gym
import pickle

env = gym.make('Breakout-v0')

num_episodes = 2
num_steps = 50

frames = []

for episode in range(num_episodes):
    obs = env.reset()
    for time_step in range(num_steps):
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        frames.append(next_obs)
        if done:
            break

pickle.dump(frames, open('frames.pkl', 'w'))


# If you want to save as images, use Image module or cv2"


