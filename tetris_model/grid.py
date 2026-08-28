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



def get_state_vector(env):
    raw_board = env.unwrapped._board.copy()
    raw_board[raw_board == 239] = 0
    board_binary = (raw_board != 0).astype(float)
    board_flat = board_binary.flatten()

    current_piece_onehot = np.zeros(len(piece_table))
    current_piece_str = env.unwrapped._current_piece
    if current_piece_str is not None:
        current_piece_onehot[piece_table.index(current_piece_str)] = 1

    next_piece_onehot = np.zeros(len(piece_table))
    next_piece_str = env.unwrapped._next_piece
    if next_piece_str is not None:
        next_piece_onehot[piece_table.index(next_piece_str)] = 1

    state_vector = np.concatenate([board_flat, current_piece_onehot, next_piece_onehot])
    return state_vector.reshape(1,-1)
    
