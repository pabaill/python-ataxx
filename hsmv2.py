import numpy as np
from tqdm import tqdm
import ataxx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import random
import copy

import ataxx.players

def default_eval(board: ataxx.Board, turn=None):
    black, white, _, _ = board.count()
    if turn and turn == ataxx.BLACK or (board.turn == ataxx.BLACK):
        return black - white
    else:
        return white - black
    

def improved_eval(board: ataxx.Board, move: ataxx.Move, gamma=0.9, root=True):
    s1 = 10
    s2 = 4
    s3 = 7
    s4 = 4

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

    # Consider long-term rewards (future board evaluation)
    long_term_score = 0
    # Check the board after the move to account for future rewards
    if root:
        for next_move in new_board.legal_moves():
            # Should this be negative? Negative scores for putting opponent in good position right?
            long_term_score -= improved_eval(new_board, next_move, gamma=gamma, root=False)

    Si += gamma * long_term_score  # Incorporate future rewards
    
    return Si

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim=7):
        super(PolicyNetwork, self).__init__()
        self.input_size = (action_dim + 4) ** 2
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim * action_dim)

        # Apply He initialization for weights
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

        # Initialize biases to zero (or small positive values if needed)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def select_action(self, board: ataxx.Board, epsilon=None):
        """ Select an action based on the policy network using softmax. State is tensor-ified board"""
        if board.must_pass() or board.gameover():
            return ataxx.Move.null()
        state = self.board_to_tensor(board)
        legal_moves = board.legal_moves()
        # print([f"{(m.fr_x, m.fr_y)} to {(m.to_x, m.to_y)}" for m in legal_moves])
        logits = self(state)
        action_probs = torch.softmax(logits, dim=-1)
        
        # Create a mask to set illegal start positions for our action
        legal_action_probs = torch.zeros_like(action_probs)
        for move in legal_moves:
            # DON'T PASS IF WE DON'T HAVE TO!
            if not (move.fr_x == move.to_x and move.fr_y == move.to_y):
                action_idx_from = move.fr_x * 7 + move.fr_y
                legal_action_probs[action_idx_from] = action_probs[action_idx_from]
        
        # Epsilon-greedy action selection
        if epsilon and random.random() < epsilon:  # Exploration
            start_action_idx = torch.multinomial(legal_action_probs, 1).item()
        else:  # Exploitation
            start_action_idx = torch.argmax(legal_action_probs).item()
        
        # Convert the action index to (x1, y1)
        x1, y1 = self.index_to_coords(start_action_idx)
        # print(x1, y1)

        # Create a mask to set illegal start positions for our action
        legal_action_probs = torch.zeros_like(action_probs)
        for move in legal_moves:
            if not (move.to_x == x1 and move.to_y == y1) and move.fr_x == x1 and move.fr_y == y1:
                # print(move.to_x, move.to_y)
                action_idx_to = move.to_x * 7 + move.to_y
                legal_action_probs[action_idx_to] = action_probs[action_idx_to]
        
        if epsilon and random.random() < epsilon:  # Exploration for second action
            end_action_idx = torch.multinomial(legal_action_probs, 1).item()
        else:  # Exploitation for second action
            end_action_idx = torch.argmax(legal_action_probs).item()
        x2, y2 = self.index_to_coords(end_action_idx)

        return ataxx.Move(x1, y1, x2, y2)
    
    @staticmethod
    def board_to_tensor(board):
        """ Convert the Ataxx game board (list of lists) to a PyTorch tensor. """
        board_np = np.array(board._board, dtype=np.float32)
        board_tensor = torch.tensor(board_np)
        
        # Optionally, flatten the tensor for input to neural network
        return board_tensor.flatten()
    
    @staticmethod
    def index_to_coords(index):
        return index // 7, index % 7
    
def generate_random_board():
    board = ataxx.Board()
    n_moves = random.randint(1, 25)
    for i in range(n_moves):
        player = random.randint(0, 2)
        if player == 0:
            move = ataxx.players.alphabeta(board, float('-inf'), float('inf'), 2)
        elif player == 1:
            move = ataxx.players.negamax(board, 2)
        else:
            move = ataxx.players.greedy(board)
        board.makemove(move)
        if board.gameover():
            board.undo()
            i -= 1
    if board.turn != ataxx.WHITE:
        board.undo()
    return board


def get_best_move(board: ataxx.Board):
    """Evaluate all legal moves with a two step lookahead and return the best possible move based on the evaluation."""
    best_move = None
    best_score = -float('inf')
    one_step_best_score = -float('inf')
    turn = board.turn
    
    for move in board.legal_moves():
        immediate_score = improved_eval(board, move)
        board.makemove(move)
        for next_move in board.legal_moves():
            score = improved_eval(board, next_move)  # Use default_eval as the heuristic for evaluating moves
            if score > best_score:
                best_score = score
                one_step_best_score = immediate_score
                best_move = move
        board.undo()
            
            
    return best_move, one_step_best_score

def train_policy_network(model_name, batch_size=32, action_dim=7, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=0.995, lr_start=0.001, lr_decay=0.7):
    model = PolicyNetwork(action_dim=action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr_start, weight_decay=0.01)
    epsilon = epsilon_start

    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

    board_fens = []
    for i in tqdm(range(200)):
        b = generate_random_board()
        board_fens.append(b.get_fen())
    
    # Shuffle the data
    random.shuffle(board_fens)

    batches = []
    for i in range(0, len(board_fens), batch_size):
        batch = board_fens[i:i + batch_size]
        batches.append(batch)

    # List to store loss values for plotting
    loss_history = []

    for epoch in range(200):
        epoch_loss = 0  # Track loss for this epoch
        n_valid_states = 0
        for batch in tqdm(batches):
            board_batch_fens = []
            actions = []
            score_batch = []
            for board_fen in batch:
                board = ataxx.Board(board_fens[i])  # Re-create the board from FEN
                move = model.select_action(board, epsilon=epsilon)  # Get action from the model
                x1, y1, x2, y2 = move.fr_x, move.fr_y, move.to_x, move.to_y
                if x1 is not None:
                    board_batch_fens.append(board_fen)
                    actions.append(move)
                    score_batch.append(improved_eval(board, move))

            # Get the scores for the selected actions
            selected_scores_tensor = torch.tensor(score_batch, dtype=torch.float32, requires_grad=True)

            # Loss function: we want to maximize the score, so we minimize the negative of the score
            loss = -selected_scores_tensor.mean()  # Take the mean loss over the batch
            epoch_loss += loss.item()

            # Backpropagate and update the model
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the model parameters
            optimizer.step()

            n_valid_states += len(board_batch_fens)

        # Update epsilon for exploration-exploitation tradeoff
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        # scheduler.step()  # Update learning rate
        print(f"Epoch {epoch}: {epoch_loss / n_valid_states}")


        # Record loss for plotting
        if n_valid_states > 0:
            loss_history.append(epoch_loss / n_valid_states)

    # Plot the training loss
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # Save final model
    torch.save(model.state_dict(), f'{model_name}.pth')


