import numpy as np
import random
import copy
import ataxx
from tqdm import tqdm
from heapq import nlargest

ROLLOUT_DEPTH = 5

BEAM_SIZE = 3

def improved_eval(board: ataxx.Board, move: ataxx.Move):
    s1 = 1
    s2 = 0.4
    s3 = 0.7
    s4 = 0.4

    color = board.turn
    opponent_color = ataxx.WHITE if color == ataxx.BLACK else ataxx.BLACK

    # Check if move is a pass
    if move == ataxx.Move.null():
        return 0

    from_x, from_y = move.fr_x, move.fr_y
    to_x, to_y = move.to_x, move.to_y

    dx = abs(from_x - to_x)
    dy = abs(from_y - to_y)

    # Determine if the move is a jump move
    is_jump_move = max(dx, dy) == 2

    # Make a copy of the board and apply the move
    new_board = copy.deepcopy(board)
    new_board.makemove(move)

    # Define adjacent positions
    adjacent_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]

    # Count enemy stones taken over
    enemy_stones_taken = 0
    for dx_offset, dy_offset in adjacent_offsets:
        nx, ny = to_x + dx_offset, to_y + dy_offset
        if board.get(nx, ny) == opponent_color:
            enemy_stones_taken += 1
    Si = s1 * enemy_stones_taken

    # Count own stones around the target square after the move
    own_stones_around_target = 0
    for dx_offset, dy_offset in adjacent_offsets:
        nx, ny = to_x + dx_offset, to_y + dy_offset
        if board.get(nx, ny) == color:
            own_stones_around_target += 1
    Si += s2 * own_stones_around_target

    # Add s3 if the move is not a jump move
    if not is_jump_move:
        Si += s3

    # Subtract s4 times own stones around the source square if it's a jump move
    if is_jump_move:
        own_stones_around_source = 0
        for dx_offset, dy_offset in adjacent_offsets:
            nx, ny = from_x + dx_offset, from_y + dy_offset
            if board.get(nx, ny) == color:
                own_stones_around_source += 1
        Si -= s4 * own_stones_around_source
    
    Si = max(0, Si)

    return Si

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.reward = 0
        self.legal_moves = list(board.legal_moves())
    
    def is_fully_expanded(self):
        # return len(self.children) == len(self.legal_moves)
        return len(self.children) >= min(len(self.legal_moves), BEAM_SIZE)
    
    def best_child(self, exploration_weight=1.4):
        # """ Select the child with the highest UCB1 value """
        if not self.children:
            return None
        ucb_values = [
            child.reward / (child.visits + 1e-6) + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(ucb_values)]

def mcts(board, simulations=1000):
    root_node = MCTSNode(board)
    
    for _ in tqdm(range(simulations)):
        node = root_node

        # Selection phase: Traverse the tree until a leaf node is found
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion phase: If not fully expanded, expand the node using beam search
        if not node.is_fully_expanded():
            # Evaluate all legal moves
            move_scores = [(move, improved_eval(node.board, move)) for move in node.legal_moves]
            # Sort moves by their evaluation scores in descending order
            move_scores.sort(key=lambda x: x[1], reverse=True)
            # Select the top `beam_width` moves
            top_moves = [move for move, _ in move_scores[:BEAM_SIZE]]
            
            for move in top_moves:
                # For each top move, expand the tree
                new_board = copy.deepcopy(node.board)
                new_board.makemove(move)
                child_node = MCTSNode(new_board, parent=node, move=move)
                node.children.append(child_node)
        
        # Simulation phase: Use improved_eval to simulate a game and return the winner
        evaluation_score = simulate_with_eval(node.board)

        # Backpropagation phase: Update the node values along the path
        while node:
            node.visits += 1
            node.reward += evaluation_score
            node = node.parent
    
    # Select the move with the highest visit count from the root
    best_child = root_node.best_child(exploration_weight=0)
    return best_child.move

def simulate_with_eval(board):
    """ Use improved_eval to simulate a game and return the winner, considering only the top BEAM_SIZE moves. """
    board = copy.deepcopy(board)
    color = board.turn
    
    # While not reaching the maximum rollout depth (ROLLOUT_DEPTH)
    for i in range(ROLLOUT_DEPTH):
        # Generate legal moves and evaluate them using improved_eval
        moves = board.legal_moves()
        
        # If no legal moves, break the simulation
        if not moves:
            break
        
        # Evaluate the moves and keep track of the top BEAM_SIZE moves using nlargest
        move_scores = [(move, improved_eval(board, move)) for move in moves]
        
        # Use nlargest to get the top BEAM_SIZE moves (faster than sorting)
        top_moves = nlargest(BEAM_SIZE, move_scores, key=lambda x: x[1])
        
        # Select one move from the top moves based on the normalized evaluation scores
        top_moves_scores = np.array([score for _, score in top_moves])
        z = top_moves_scores.sum()
        if z == 0 or np.isnan(z):
            break
        normalized_scores = top_moves_scores / z
        make_move = np.random.choice([move for move, _ in top_moves], p=normalized_scores)
        
        # Make the move on the board
        board.makemove(make_move)
    
    # Return the winner (1 for the current player, -1 for opponent, 0 for draw)
    black, white, _, _ = board.count()
    if black > white:
        return 1 if color == ataxx.BLACK else -1  # Current player wins
    elif white > black:
        return 1 if color == ataxx.WHITE else -1  # Current player wins
    return 0  # Draw


def get_best_move(board: ataxx.Board, simulations=1000):
    """Use MCTS to select the best move, integrating the improved_eval function."""
    new_board = copy.deepcopy(board)
    best_move = mcts(new_board, simulations=simulations)
    return best_move