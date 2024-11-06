import ataxx
import sys
import ataxx.players

def play_solo():
    fen = input("FEN: ")
    if fen == "":
        fen = "startpos"
    board = ataxx.Board(fen)

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

def alphabeta(board, alpha, beta, depth, root=True):
    if depth == 0:
        black, white, _, _ = board.count()
        if board.turn == ataxx.BLACK:
            return black - white
        else:
            return white - black

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
def play_alphabeta(depth=2, opponent='alphabeta'):
    board = ataxx.Board("startpos")
    turn_counter = 1

    while not board.gameover():
        try:
            if turn_counter % 2 == 1:
                move = alphabeta(board, float('-inf'), float('inf'), depth)
            else:
                if opponent == 'alphabeta':
                    move = opponent(board, float('-inf'), float('inf'), depth)
                else:
                    move = opponent(board)
            if board.is_legal(move):
                board.makemove(move)
            else:
                print(F"Illegal move: {move}")
        except KeyboardInterrupt:
            print("")
            break
        # print(board)
        # next = input("NEXT:")
        turn_counter += 1
    
    print(F"Result: {board.result()}")
    return board.result()



def main():
    if len(sys.argv) == 0:
        print("Available modes: solo")
    mode = sys.argv[1]
    if mode == 'solo':
        play_solo()
    if mode == 'alphabeta':
        wincount = 0
        n_games = 10
        for i in range(n_games):
            result = play_alphabeta(opponent=ataxx.players.random_move)
            if result == '1-0':
                wincount+=1
        print(wincount / n_games)



if __name__ == '__main__':
    main()
