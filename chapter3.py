import gym
import numpy as np

env = gym.make("FrozenLake-v0")
env = env.unwrapped

nA = env.action_space.n
nS = env.observation_space.n

V = np.zeros(nS)
policy = np.zeros(nA)