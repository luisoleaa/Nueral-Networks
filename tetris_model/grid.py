# Have to break down the frames of the tetris game (state) so the neural network can "see"
# what is going on in the game to take into each layer of the network
# This is done using the RAM info the library exposes (current_piece, next_piece, board_height, and a raw board grid)

# board: 20 rows x 10 columns, 1 if occupied, 0 if empty → 200 values
# current_piece: one-hot encoded across the possible piece types
# next_piece: one-hot encoded across the possible piece types
import numpy as np
import gym_tetris

# all possible tetris pieces
piece_table = ['Tu','Tr','Td','Tl','Jl','Ju','Jr','Jd','Zh','Zv','O','Sh','Sv','Lr','Ld','Ll','Lu','Iv','Ih']

def binary_board(env):
    b = env.unwrapped._board.copy()
    b[b == 239] = 0
    return (b != 0).astype(np.int8) # (20,10), 1 = filled, row 0 = top

def column_heights(bb):
    rows, cols = bb.shape # rows = 20, cols = 10
    heights = np.zeros(cols, dtype=np.int32)
    for c in range(cols):
        filled = np.where(bb[:,c] == 1)[0] # checks if the block at column c and row 0
        if filled.size: # if its 1
            heights[c] = rows - filled[0]
    return heights

def count_holes(bb):
    rows, cols = bb.shape
    holes = 0
    for c in range(cols):
        seen = False
        for r in range(rows):
            if bb[r,c]: # if true then there is a block here
                seen = True
            elif seen: # if no block under the last block and there is one above
                holes += 1 # there is a whole
    return holes

# collapse the board into a "how good is this position" -> Delta shaping formula
# Φ(board) = -0.51 * aggregate_height
#            - 0.36 * holes
#            - 0.18 * bumpiness

PLACEMENT_FEATURE_NAMES = ["agg_height", "holes", "bumpiness"]

def board_features(bb):
    # [aggregate_height, holes, bumpiness]. board_potential and
    # placement_score both score off this same array so the GA weights
    # line up with the same features everywhere, no duplicate math.
    height = column_heights(bb)
    holes = count_holes(bb)
    bump = int(np.abs(np.diff(height)).sum())
    return np.array([height.sum(), holes, bump], dtype=np.float64)

DEFAULT_POTENTIAL_WEIGHTS = np.array([-0.51, -0.36, -0.18])

def board_potential(bb, weights=None):
    if weights is None:
        weights = DEFAULT_POTENTIAL_WEIGHTS
    return float(np.dot(weights, board_features(bb)))

# genome / placement-scoring convention: [w_lines, w_agg_height, w_holes, w_bumpiness]
DEFAULT_PLACEMENT_WEIGHTS = np.array([20.0, -0.51, -0.36, -0.18])

def placement_score(new_board, n_cleared, weights):
    # used by both heuristic_bot's live search and evolve_bot's offline
    # simulator so they never drift out of sync with each other
    w_lines, feat_weights = weights[0], weights[1:]
    return w_lines * n_cleared + float(np.dot(feat_weights, board_features(new_board)))


def row_progress_score(bb):
    # sum of (filled cells per row)^2. rewards stacking fill into rows that
    # are already pretty full instead of spreading pieces evenly -- board_potential
    # doesnt care about this, it only looks at height/holes/bumpiness
    row_fill = bb.sum(axis=1).astype(np.float64)
    return float(np.sum(row_fill ** 2))


def get_state_vector(env):
    u = env.unwrapped
    bb = binary_board(env)
    board_flat = bb.flatten()
    holes = count_holes(bb)
    height = column_heights(bb).astype(np.float64) 
    bump = np.abs(np.diff(height)) 


    current_piece = u._current_piece
    current_piece_onehot = np.zeros(len(piece_table))
    px, py = (float(u.ram[0x40]), float(u.ram[0x41]))
    if current_piece is not None:
        current_piece_onehot[piece_table.index(current_piece)] = 1

    next_piece_onehot = np.zeros(len(piece_table))
    next_piece = u._next_piece
    if next_piece is not None:
        next_piece_onehot[piece_table.index(next_piece)] = 1
        
    engineered = np.concatenate([
        height / 20.0,                     # 10  per-column heights
        bump / 20.0,                  # 9   bumpiness
        [height.sum()  / 200.0],           
        [height.max()  / 20.0],            
        [holes / 50.0],
        [px / 10.0, py / 20.0]       # 2 active pieces
    ])
    # total width should be 262

    state_vector = np.concatenate([board_flat, current_piece_onehot, next_piece_onehot, engineered])
    return state_vector.reshape(1,-1)

