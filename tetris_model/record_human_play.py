# record_human_play.py -- lets you record your own keyboard play as
# (state, action) pairs, same format as heuristic_bot.py's demos (see
# generate_demos.py / pretrain_bc.py), in case you want to mix them in.
#
# why not just use play_human's callback directly? because
# nes_py.app.play_human.play_human(env, callback) only calls the callback
# AFTER env.step() runs -- by then the RAM already reflects the post-step
# state, so you can't read get_state_vector(env) for the state the action
# was actually chosen from anymore. fixed by capturing the state inside a
# step() override, before it delegates to the real step.
#
# note on granularity: this records every raw frame (no k=4 frame skip),
# since play_human drives the emulator at native speed, unlike
# heuristic_bot's demos which are one entry per macro-step. doesnt matter for
# BC since it just learns state->action regardless of timing, but it does
# mean human demos end up way denser per unit of gameplay than bot demos.
# not a bug, just dont be confused when the ratios look different.
#
# usage: run this, play with the keyboard in the window that pops up, hit
# Escape when youre done. running it again later appends onto the same
# human_demos.npz instead of overwriting, so you can record over multiple
# sessions.

import os
import numpy as np
import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from nes_py.app.play_human import play_human
from grid import get_state_vector

OUT_PATH = "human_demos.npz"


class RecordingJoypad(JoypadSpace):
    # same as JoypadSpace but grabs (state, action) on every step before the
    # underlying env actually advances

    def __init__(self, env, movement):
        super().__init__(env, movement)
        self.states = []
        self.actions = []

    def step(self, action):
        self.states.append(get_state_vector(self)[0])
        self.actions.append(action)
        return super().step(action)


def main():
    env = gym_tetris.make('TetrisA-v1')
    env = RecordingJoypad(env, MOVEMENT)

    print("playing -- use the keyboard in the viewer window, Escape to stop and save.")
    try:
        play_human(env)
    finally:
        new_states = np.array(env.states, dtype=np.float32)
        new_actions = np.array(env.actions, dtype=np.int64)

        if len(new_states) == 0:
            print("no steps recorded, nothing saved.")
            return

        if os.path.exists(OUT_PATH):
            existing = np.load(OUT_PATH)
            new_states = np.concatenate([existing['states'], new_states])
            new_actions = np.concatenate([existing['actions'], new_actions])

        np.savez_compressed(OUT_PATH, states=new_states, actions=new_actions)
        print(f"saved {OUT_PATH}: {len(new_states)} total (state, action) pairs")
        print("action distribution:", np.bincount(new_actions, minlength=12))


if __name__ == "__main__":
    main()
