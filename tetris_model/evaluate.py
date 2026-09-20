
# evaluate.py -- "did the network actually learn anything?"
#
# rebuilds the same architecture main.py uses, plays N games with a random
# policy (no brain at all) to get a baseline, then loads the trained weights
# and plays N more games with those. prints both side by side.
#
# if trained is clearly better (higher return, more steps, more lines) then
# training helped. if it looks about the same as random, it didnt learn
# anything yet.


import numpy as np
import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from model import Layer, Relu, softmax
from grid import get_state_vector, binary_board, board_potential, row_progress_score



from functions import load_checkpoint
from eval_viz import plot_eval
W_LINES = 5.0   # keep in sync with W_lines in main.py
GAMMA = 0.99    # keep in sync with gamma in main.py
W_TOPOUT = 3    # keep in sync with W_TOPOUT in main.py
W_ROW_PROGRESS = 0.05   # keep in sync with W_ROW_PROGRESS in main.py

env = gym_tetris.make('TetrisA-v1')
env = JoypadSpace(env, MOVEMENT)          # maps the 12 discrete actions to controller inputs

state_size = get_state_vector(env).shape[1]   # 238 (see grid.py)
num_actions = 12

Layer1, Layer2 = Layer(state_size, 128), Layer(128, 64)
Relu1, Relu2 = Relu(), Relu()
policy_head, value_head = Layer(64, num_actions), Layer(64, 1)

layers = [Layer1, Layer2, policy_head, value_head]


def policy_action(state, greedy=True):
    # one forward pass -> action (0..11). greedy=True always takes the
    # highest-prob action (deterministic, best-effort). greedy=False samples
    # from the distribution instead, use that if you want to see it explore.
    x = Relu1.forward(Layer1.forward(state))
    x = Relu2.forward(Layer2.forward(x))
    probs = softmax(policy_head.forward(x))[0]        # shape (12,), sums to 1
    return int(np.argmax(probs)) if greedy else int(np.random.choice(num_actions, p=probs))


def run_episodes(n, act_fn):
    raw_rets, shaped_rets, lens, lines, terms = [], [], [], [], []
    hist = np.zeros(num_actions)

    for _ in range(n):
        env.reset()
        done = False
        raw_ret, shaped_ret, length = 0.0, 0.0, 0
        info = {}

        phi_prev = board_potential(binary_board(env))   # same as main.py
        row_prog_prev = row_progress_score(binary_board(env))
        prev_lines = 0

        while not done:
            a = act_fn(get_state_vector(env))
            hist[a] += 1
            k = 4
            for j in range(k):
                _, r, done, info = env.step(a)
                if done:
                    break

            bb_now = binary_board(env)
            phi_now = board_potential(bb_now)
            row_prog_now = row_progress_score(bb_now)
            lines_cleared = info['number_of_lines'] - prev_lines
            prev_lines = info['number_of_lines']
            if done:
                shaped = W_LINES * lines_cleared - W_TOPOUT
            elif lines_cleared > 0:
                shaped = W_LINES * lines_cleared + (phi_now - phi_prev)
            else:
                shaped = (W_LINES * lines_cleared
                          + (phi_now - phi_prev)
                          + W_ROW_PROGRESS * (row_prog_now - row_prog_prev))
            phi_prev = phi_now
            row_prog_prev = row_prog_now

            raw_ret += r
            shaped_ret += shaped
            length += 1
            if length > 5000:
                break

        raw_rets.append(raw_ret)
        shaped_rets.append(shaped_ret)
        lens.append(length)
        lines.append(info.get('number_of_lines', 0))
        terms.append(bool(done))          # True = topped out, False = hit the step cap

    return (np.array(raw_rets), np.array(shaped_rets),
            np.array(lens), np.array(lines), hist, np.array(terms))



# --- run the comparison ----------------------------------------------------
N = 20   # games per policy. Bigger N = less noisy averages, slower to run.

# Baseline: no learning at all, pick a random action every frame.
r = run_episodes(N, lambda s: np.random.randint(num_actions))

# Load trained weights into the layers, then evaluate greedily (best-effort play).
it = load_checkpoint("Tetris_ppo.npz", layers)   # returns the iteration number stored in the file
t = run_episodes(N, lambda s: policy_action(s, greedy=True))

# --- report --------------------------------------------------------------
print(f"checkpoint iteration: {it}")
for name, (raw, shaped, ln, li, h, term) in [("random", r), ("trained", t)]:
    print(f"{name:8}  raw {raw.mean():7.1f}   shaped {shaped.mean():8.1f} +/- {shaped.std():6.1f}   "
          f"steps {ln.mean():6.0f}   lines {li.mean():4.1f}   topped-out {term.mean()*100:3.0f}%")

# If these counts are all piled on one or two actions, the policy collapsed
# (it "learned" a degenerate habit rather than how to play).
print("trained action counts:", t[4].astype(int))

env.close()

# side-by-side matplotlib comparison (saves eval_comparison.png and opens a window)
plot_eval(r, t, it, n=N)

# how to read this:
#   trained clearly > random -> training helped
#   trained ~= random -> didnt really learn much (too few iters? no advantage
#                         normalization? bad LR? reward too sparse?)
#   action counts piled onto 1-2 actions -> policy collapsed, still a problem
#                         even if the return number looks ok
#
# to actually WATCH it play instead of just scoring it: change greedy=True to
# greedy=False above, and add env.render() inside run_episodes's while loop
# (its gonna be slow)
