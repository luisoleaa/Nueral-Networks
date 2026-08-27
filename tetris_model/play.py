import gymnasium as gym
import gym_tetris
from nes_py.play import play_human

env = gym.make('TetrisA-v0', render_mode=None)
play_human(env)
