import ataxx
import sys
import ataxx.players
import numpy as np
from ataxx import FEN_4CORNERS, FEN_4SIDES, FEN_CENTERX, FEN_EMPTY, FEN_ISLAND, FEN_STARTPOS
from tqdm import tqdm
import torch
import hsmv2
import mcts
import matplotlib.pyplot as plt

BLACK, WHITE, GAP, EMPTY = 0, 1, 2, 3

N_MOVES_EXAMPLE = 35
POSSIBLE_BOARDS = [FEN_4CORNERS, FEN_4SIDES, FEN_CENTERX, FEN_EMPTY, FEN_ISLAND, FEN_STARTPOS]
PLAYER_TYPES = [ataxx.players.alphabeta, ataxx.players.greedy, ataxx.players.negamax]
N_TRAINING_EXAMPLES = 1000
TRAIN_DEPTH = 2

def load_end_data():
    with open("endgame.txt", "r") as datafile:
        data = datafile.readlines()
        for fen in data:
            fen = fen.strip()
            board = ataxx.Board(fen)
            print(board)

def gen_end_examples():
    with open("endgame.txt", 'w+') as outfile:
        for i in tqdm(range(N_TRAINING_EXAMPLES)):
            board = ataxx.Board(np.random.choice(POSSIBLE_BOARDS))
            p1 = np.random.choice(PLAYER_TYPES)
            p2 = np.random.choice(PLAYER_TYPES)
            for m in range(N_MOVES_EXAMPLE):
                for p in [p1, p2]:
                    if p == ataxx.players.alphabeta:
                        move = p(board, float('-inf'), float('inf'), TRAIN_DEPTH)
                    elif p == ataxx.players.negamax:
                        move = p(board, TRAIN_DEPTH)
                    elif p == ataxx.players.greedy:
                        move = p(board)
                if board.gameover():
                    break
                board.makemove(move)
            # print(board)
            if board.gameover():
                # We went too far, retry generating
                m -= 1
            else:
                fen = board.get_fen()
                outfile.write(fen + "\n")

def play_solo():
    fen = input("FEN: ")
    board_dim = int(input("BOARD SIZE: "))
    if fen == "":
        fen = "startpos"
    board = ataxx.Board(fen, board_dim=board_dim)
    print(board)

    while not board.gameover():
        print("\n\n\n")
        print(F"FEN: {board.get_fen()}")
        print(board)
        try:
            print(f'Legal moves: {[(m.to_x, m.to_y) for m in board.legal_moves()]}')
            move_string = input("Move: ")
            move = ataxx.Move.from_san(move_string)
            if board.is_legal(move):
                board.makemove(move)
            else:
                print(F"Illegal move: {move}")
        except KeyboardInterrupt:
            print("")
            break

    print(F"Result: {board.result()}")

def default_eval(board: ataxx.Board):
    black, white, _, _ = board.count()
    if board.turn == ataxx.BLACK:
        return black - white
    else:
        return white - black

"""
Evaluation function that favors pieces placed closer to the center of the board
"""
def weight_center_eval_function(board: ataxx.Board):
    player = board.turn
    points = default_eval(board)
    # Define the center coordinates
    center_x, center_y = (board.board_dim - 1) // 2, (board.board_dim - 1) // 2
    
    # Iterate over the grid and calculate the Manhattan distance to the center
    for i in range(board.board_dim):
        for j in range(board.board_dim):
            # Manhattan distance from the current cell (i, j) to the center
            if board.get(i, j) == player:
                distance = abs(i - center_x) + abs(j - center_y)
            
                # Assign points inversely proportional to the distance
                # We can use the formula `points = max_points - distance`
                # where max_points is the maximum possible distance, which is n-1 for the corners
                max_distance = (board.board_dim - 1) * 2
                p = max(0, max_distance - distance)  # Ensure no negative points
                
                points += p
    return points

"""
Evaluation function that discourages "L" and "U" shapes
"""
def keep_clustered_eval_function(board: ataxx.Board):
    points = default_eval(board)
    player = board.turn

    def find_l_shapes(board: ataxx.Board):
        rows, cols = len(board), len(board[0])
        l_shapes = []

        # Loop through the grid and check for L shapes
        for i in range(1, board.board_dim - 1):
            for j in range(1, board.board_dim - 1):
                # Check for L shapes (both horizontal and vertical parts)
                
                # 1. L shape - bottom-left corner (┐)
                if board[i][j] == 1 and board[i-1][j] == 1 and board[i][j-1] == 1:
                    l_shapes.append(((i, j), "┐"))
                
                # 2. L shape - top-left corner (└)
                if board[i][j] == 1 and board[i-1][j] == 1 and board[i][j+1] == 1:
                    l_shapes.append(((i, j), "└"))
                
                # 3. L shape - top-right corner (┌)
                if board[i][j] == 1 and board[i+1][j] == 1 and board[i][j-1] == 1:
                    l_shapes.append(((i, j), "┌"))
                
                # 4. L shape - bottom-right corner (┌)
                if board[i][j] == 1 and board[i+1][j] == 1 and board[i][j+1] == 1:
                    l_shapes.append(((i, j), "└"))

        return l_shapes

    def find_u_shapes(board: ataxx.Board):
        rows, cols = len(board), len(board[0])
        u_shapes = []

        # Loop through the grid and check for U shapes
        for i in range(1, board.board_dim - 1):
            for j in range(1, board.board_dim - 1):
                # Check for U shapes with different possible configurations.
                
                # 1. U shape - open top, with vertical sides
                if board[i][j] == 1 and board[i][j-1] == 1 and board[i][j+1] == 1 and board[i-1][j] == 1:
                    u_shapes.append(((i, j), "U"))

        return u_shapes

    return points - 5 * len(find_l_shapes(board)) - 5 * len(find_u_shapes(board))


def play_mcts(opponent="alphabeta", board_dim=7, depth=2):
    board = ataxx.Board(board_dim=board_dim)
    turn_counter = 1

    game_history = []

    while not board.gameover():
        try:
            if turn_counter % 2 == 1:
                # Opponent goes first
                if opponent == 'alphabeta':
                    move = ataxx.players.alphabeta(board, float('-inf'), float('inf'), depth)
                else:
                    move = opponent(board)
            else:
                move = mcts.get_best_move(board, simulations=100)
            if board.is_legal(move):
                board.makemove(move)
            else:
                print(f"Illegal move: {move}")
        except KeyboardInterrupt:
            print("")
            break
        game_history.append(-board.score())
        turn_counter += 1
        print(board)
    
    print(f"Result: {board.result()}")
    plt.plot(game_history)
    plt.xlabel('Turn')
    plt.ylabel('Net Pieces')
    plt.show()
    return board.result()



def play_hierarchical_softmax_agent(agent, depth=2, opponent='alphabeta', board="startpos", board_dim=7):
    board = ataxx.Board(board, board_dim=board_dim)
    turn_counter = 1

    while not board.gameover():
        try:
            if turn_counter % 2 == 1:
                # Opponent goes first
                if opponent == 'alphabeta':
                    move = ataxx.players.alphabeta(board, float('-inf'), float('inf'), depth)
                else:
                    move = opponent(board)
            else:
                # Your Hierarchical Softmax agent's move
                move = agent.select_action(board)
            if board.is_legal(move):
                board.makemove(move)
            else:
                print(f"Illegal move: {move}")
        except KeyboardInterrupt:
            print("")
            break
        turn_counter += 1
        print(board)
    
    print(f"Result: {board.result()}")
    return board.result()


def alphabeta(board, alpha, beta, depth, root=True, evalFn=default_eval):
    if depth == 0:
        return evalFn(board)

    best_move = None

    for move in board.legal_moves():
        board.makemove(move)
        score = -alphabeta(board, -beta, -alpha, depth-1, root=False)

        if score > alpha:
            alpha = score
            best_move = move
        if score >= beta:
            score = beta
            board.undo()
            break

        board.undo()

    if root:
        return best_move
    else:
        return alpha
    
# Two alpha-beta players play against each other
def play_alphabeta(depth=2, opponent='alphabeta', board="startpos", board_dim=7, evalFn=default_eval):
    board = ataxx.Board(board, board_dim=board_dim)
    turn_counter = 1

    while not board.gameover():
        try:
            if turn_counter % 2 == 1:
                # Opponent goes first because if we go first we can win easily
                if opponent == 'alphabeta':
                    move = ataxx.players.alphabeta(board, float('-inf'), float('inf'), depth)
                else:
                    move = opponent(board)
            else:
                move = alphabeta(board, float('-inf'), float('inf'), depth, evalFn=evalFn)
            if board.is_legal(move):
                board.makemove(move)
            else:
                print(F"Illegal move: {move}")
        except KeyboardInterrupt:
            print("")
            break
        turn_counter += 1
    
    print(F"Result: {board.result()}")
    # print(board)
    return board.result()



def main():
    if len(sys.argv) == 0:
        print("Available modes: solo")
    mode = sys.argv[1]
    if mode == 'make_data':
        gen_end_examples()
    if mode == 'read_data':
        load_end_data()
    if mode == 'solo':
        play_solo()
    if mode == 'alphabeta':
        # board = input("BOARD FEN: ")
        board_size = int(input("BOARD SIZE: "))
        wincount = 0
        n_games = 100
        for i in range(n_games):
            result = play_alphabeta(opponent='alphabeta', board_dim=board_size, evalFn=keep_clustered_eval_function)
            if result == '1-0':
                wincount+=1
        print(wincount / n_games)
    if mode == 'train_hsmv2':
        hsmv2.train_policy_network(sys.argv[2])
    if mode == 'hsmv2':
        agent = hsmv2.PolicyNetwork(action_dim=7)  # Reinitialize the model
        agent.load_state_dict(torch.load(sys.argv[2]))  # Load the weights
        agent.eval()  # Set to evaluation mode if not continuing training
        board_size = 7
        wincount = 0
        n_games = 1
        for i in range(n_games):
            result = play_hierarchical_softmax_agent(agent, opponent='alphabeta', board_dim=board_size)
            if result == '1-0':
                wincount+=1
        print(wincount / n_games)
    if mode == 'mcts':
        wincount = 0
        n_games = 1
        for i in range(n_games):
            result = play_mcts(opponent=ataxx.players.greedy)
            if result == '1-0':
                wincount += 1
        print(wincount / n_games)




if __name__ == '__main__':
    main()
