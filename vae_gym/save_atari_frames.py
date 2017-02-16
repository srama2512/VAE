import gym
import pickle
import Image
import numpy as np

# Extract Y channel from RGB, resize to 84x84 and return
def toNatureDQNFormat(frame) :
	return np.array(Image.fromarray(frame).convert('YCbCr').resize((84,84),Image.BILINEAR))[:,:,0]/np.float32(255.0)

env = gym.make('Breakout-v0')

num_episodes = 5
num_steps = 1000

frames = []

for episode in range(num_episodes):
    obs = env.reset()
    for time_step in range(num_steps):
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        frames.append(toNatureDQNFormat(next_obs))
        if done:
            break

pickle.dump(frames, open('vae_gym/frames.pkl', 'w'))
#Image.fromarray(toNatureDQNFormat(frames[0])).show()


# If you want to save as images, use Image module or cv2"


