# generate_demos.py -- run the heuristic bot with recording on and save the
# resulting (state, action) pairs for behavior-cloning pretraining.

import numpy as np
from heuristic_bot import run_bot_episode, make_env

N_EPISODES = 10
MAX_PIECES = 500
OUT_PATH = "bc_demos.npz"

env = make_env()
all_states, all_actions = [], []
total_lines = 0

for i in range(N_EPISODES):
    result = run_bot_episode(env, record=True, max_pieces=MAX_PIECES)
    all_states.extend(result['states'])
    all_actions.extend(result['actions'])
    total_lines += result['lines']
    print(f"episode {i}: pieces={result['pieces']:4d}  lines={result['lines']:4d}  "
          f"demo_steps={result['demo_steps']:5d}  running_total={len(all_states)}", flush=True)

env.close()

states = np.array(all_states, dtype=np.float32)
actions = np.array(all_actions, dtype=np.int64)
np.savez_compressed(OUT_PATH, states=states, actions=actions)

print(f"\nsaved {OUT_PATH}: {states.shape[0]} (state, action) pairs, "
      f"{total_lines} total lines cleared across {N_EPISODES} episodes")
print("action distribution:", np.bincount(actions, minlength=12))
