import gym_tetris
from nes_py.app.play_human import play_human

env = gym_tetris.make('TetrisA-v0')
play_human(env)
state, reward, done, info = env
print(reward)
