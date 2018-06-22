import gym
import numpy as np
import ccxt

env = gym.make("CartPole-v0")
obs = env.reset()
print(obs)
img = env.render(mode= "rgb_array" )
print(img.shape)
print(env.action_space)

action = 1
obs,reward,done,info = env.step(action)
print(obs)
print(reward)
print(info)

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000): # 1000 steps max, we don't want to run forever
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print(np.mean(totals))