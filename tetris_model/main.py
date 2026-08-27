import gymnasium as gym
import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np

env = gym.make('TetrisA-v3', render_mode = 'human')
env = JoypadSpace(env, MOVEMENT)

observation, info = env.reset(seed = 123)
terminated = False
truncated = False

for step in range(5000):
    if terminated or truncated:
        observation, info = env.reset(seed = 123)
        terminated = False
        truncated = False
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample(),)
    frame = env.render()

env.close()