# heuristic_bot.py -- scripted "good" tetris player, mostly built so we could
# generate demo data for behavior cloning, but it ended up being the only
# thing in this project that actually plays well lol
#
# the placement search runs entirely offline in numpy against a snapshot of
# the locked board (no emulator interaction so no crash risk from nes_py).
# once it picks the best (rotation, column) it touches the real emulator to
# actually execute the placement, recording (state, action) pairs as it goes.
#
# PIECE_SHAPES were captured empirically, not typed from memory -- for each
# of the 19 orientations in gym_tetris's _PIECE_ORIENTATION_TABLE i had a
# calibration script let that orientation lock naturally and recorded the 4
# cells relative to (ram[0x41], ram[0x40]), which turns out to always be one
# of the pieces own occupied cells (the pivot). every shape below is a
# connected 4-cell tetromino and each family forms a clean 90 degree cycle --
# checked by hand against normal tetromino geometry.
#
# how actions actually behave (also figured out empirically, not from docs):
#   - rotation (A) is edge triggered. one macro-step of A = one rotate, it
#     does NOT repeat if you just hold it down.
#   - left/right follows NES DAS -- a fresh direction shifts 1 column after a
#     short lag, then needs a bunch more held frames before it repeats again.
#     so movement has to be re-issued one macro-step at a time and checked
#     against ram[0x40], can't just assume it worked.
#   - ram[0x40] = piece column, ram[0x41] = piece row.

import numpy as np
import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from grid import get_state_vector, binary_board, board_potential, placement_score, DEFAULT_PLACEMENT_WEIGHTS

NOOP_IDX  = MOVEMENT.index(['NOOP'])
A_IDX     = MOVEMENT.index(['A'])
LEFT_IDX  = MOVEMENT.index(['left'])
RIGHT_IDX = MOVEMENT.index(['right'])
DOWN_IDX  = MOVEMENT.index(['down'])

K = 4                # frame-skip, matches main.py / evaluate.py

# (row, col) offsets of each orientation's 4 cells relative to its pivot
# cell (ram[0x41], ram[0x40]). See module docstring for how these were derived.
PIECE_SHAPES = {
    'Tu': [(-1, 0), (0, -1), (0, 0), (0, 1)],
    'Tr': [(-1, 0), (0, 0), (0, 1), (1, 0)],
    'Td': [(0, -1), (0, 0), (0, 1), (1, 0)],
    'Tl': [(-1, 0), (0, -1), (0, 0), (1, 0)],
    'Jl': [(-1, 0), (0, 0), (1, -1), (1, 0)],
    'Ju': [(-1, -1), (0, -1), (0, 0), (0, 1)],
    'Jr': [(-1, 0), (-1, 1), (0, 0), (1, 0)],
    'Jd': [(0, -1), (0, 0), (0, 1), (1, 1)],
    'Zh': [(0, -1), (0, 0), (1, 0), (1, 1)],
    'Zv': [(-1, 1), (0, 0), (0, 1), (1, 0)],
    'O':  [(0, -1), (0, 0), (1, -1), (1, 0)],
    'Sh': [(0, 0), (0, 1), (1, -1), (1, 0)],
    'Sv': [(-1, 0), (0, 0), (0, 1), (1, 1)],
    'Lr': [(-1, 0), (0, 0), (1, 0), (1, 1)],
    'Ld': [(0, -1), (0, 0), (0, 1), (1, -1)],
    'Ll': [(-1, -1), (-1, 0), (0, 0), (1, 0)],
    'Lu': [(-1, 1), (0, -1), (0, 0), (0, 1)],
    'Iv': [(-2, 0), (-1, 0), (0, 0), (1, 0)],
    'Ih': [(0, -2), (0, -1), (0, 0), (0, 1)],
}


def make_env():
    env = gym_tetris.make('TetrisA-v1')
    return JoypadSpace(env, MOVEMENT)


def step_macro(env, action, k=K):
    # holds `action` for up to k raw frames, same idea as main.py's frame skip
    info, done = {}, False
    for _ in range(k):
        _, _, done, info = env.step(action)
        if done:
            break
    return done, info


def simulate_drop(board, shape, col):
    # hard drops `shape` (offsets relative to its pivot) with the pivot at
    # `col`, against a 20x10 binary board. returns (new_board, lines_cleared),
    # or None if this orientation just cant legally go in this column
    rows, cols = board.shape

    for _, dc in shape:
        c = col + dc
        if c < 0 or c >= cols:
            return None   # this orientation doesn't fit at this column at all

    def collide(row):
        for dr, dc in shape:
            r, c = row + dr, col + dc
            if r >= rows:
                return True
            if r >= 0 and board[r, c]:
                return True
        return False

    if collide(0):
        return None   # can't even spawn here

    row = 0
    while not collide(row + 1):
        row += 1

    new_board = board.copy()
    for dr, dc in shape:
        r, c = row + dr, col + dc
        if r >= 0:
            new_board[r, c] = 1

    full_rows = np.all(new_board == 1, axis=1)
    n_cleared = int(full_rows.sum())
    if n_cleared:
        keep = new_board[~full_rows]
        new_board = np.vstack([np.zeros((n_cleared, cols), dtype=new_board.dtype), keep])

    return new_board, n_cleared


def search_best_placement(env, weights=None):
    # brute force search over every (rotation, column) for the current piece
    if weights is None:
        weights = DEFAULT_PLACEMENT_WEIGHTS
    u = env.unwrapped
    spawn_name = u._current_piece
    for _ in range(10):   # RAM briefly reports no piece right after a lock; wait it out
        if spawn_name is not None:
            break
        step_macro(env, NOOP_IDX)
        spawn_name = u._current_piece
    board = binary_board(env)
    family = spawn_name[0]

    best = None   # (score, name, col)
    for name, shape in PIECE_SHAPES.items():
        if name[0] != family:
            continue
        for col in range(board.shape[1]):
            result = simulate_drop(board, shape, col)
            if result is None:
                continue
            new_board, n_cleared = result
            score = placement_score(new_board, n_cleared, weights)
            if best is None or score > best[0]:
                best = (score, name, col)

    return best[1], best[2]   # target orientation name, target column


def execute_placement(env, target_name, target_col, states_out=None, actions_out=None):
    # plays out the chosen placement for real, recording (state, action) pairs
    u = env.unwrapped

    def do(action):
        if states_out is not None:
            states_out.append(get_state_vector(env)[0])
            actions_out.append(action)
        return step_macro(env, action)

    # rotate toward target_name one press at a time, verifying against the
    # live orientation rather than assuming a fixed press count. A/left/right
    # are edge-triggered with no auto-repeat unless the button is genuinely
    # released first -- so every press is followed by a NOOP macro-step to
    # force a real release before the next one.
    for _ in range(4):
        if u._current_piece == target_name:
            break
        done, info = do(A_IDX)
        if done:
            return done, info
        done, info = do(NOOP_IDX)
        if done:
            return done, info

    # steer toward target_col, re-checking real position each step (DAS-safe)
    for _ in range(15):
        px = u.ram[0x40]
        if px == target_col:
            break
        done, info = do(RIGHT_IDX if px < target_col else LEFT_IDX)
        if done:
            return done, info
        done, info = do(NOOP_IDX)
        if done:
            return done, info
        if u.ram[0x40] == px:
            break   # stalled (wall / blocked) -- close enough, proceed to drop

    # soft-drop to lock
    piece_before = u._current_piece
    done, info = {}, {}
    for _ in range(25):
        done, info = do(DOWN_IDX)
        if done or u._current_piece != piece_before:
            break
    return done, info


def run_bot_episode(env, record=False, max_pieces=500, weights=None):
    env.reset()
    prev_lines = 0
    states, actions = ([], []) if record else (None, None)
    steps = 0
    done = False
    info = {}

    for steps in range(1, max_pieces + 1):
        target_name, target_col = search_best_placement(env, weights=weights)
        done, info = execute_placement(env, target_name, target_col, states, actions)
        prev_lines = info.get('number_of_lines', prev_lines)
        if done:
            break

    return {
        'lines': prev_lines,
        'pieces': steps,
        'demo_steps': len(states) if states is not None else 0,
        'states': states,
        'actions': actions,
    }


if __name__ == "__main__":
    env = make_env()

    N = 5
    print(f"running {N} bot episodes (no recording) as a quick sanity check...")
    for i in range(N):
        result = run_bot_episode(env, record=False, max_pieces=400)
        print(f"episode {i}: pieces={result['pieces']:4d}  lines={result['lines']}")

    env.close()
