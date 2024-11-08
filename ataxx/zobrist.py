import ataxx
import random


turn = 0x2e98304a94e1000d

# Maximum size of the board, assuming `board_dim` could go up to 10 for example
MAX_BOARD_DIM = 99

# Generate random hashes for each piece position
def generate_piece_hashes(board_dim):
    return [random.getrandbits(64) for _ in range(board_dim * board_dim)]

# Piece array will now dynamically generate hashes based on the board size
piece = [
    generate_piece_hashes(MAX_BOARD_DIM),  # For BLACK
    generate_piece_hashes(MAX_BOARD_DIM),  # For WHITE
]

# You can adjust the above `MAX_BOARD_DIM` based on your needs or pass `board_dim` dynamically.


def calculate_hash(b, board_dim):
    key = 0

    if b.turn == ataxx.BLACK:
        key ^= turn

    for y in range(board_dim):
        for x in range(board_dim):
            if b.get(x, y) == ataxx.BLACK:
                # For BLACK pieces, use piece[0]
                key ^= piece[0][y * board_dim + x]
            elif b.get(x, y) == ataxx.WHITE:
                # For WHITE pieces, use piece[1]
                key ^= piece[1][y * board_dim + x]

    return key

def get_turn_hash(side):
    return turn

def get_sq_hash(x, y, side, board_dim):
    return piece[side][y * board_dim + x]
