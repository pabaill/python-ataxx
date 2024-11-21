import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
from gym import spaces
import numpy as np
import ataxx
from tqdm import tqdm

import ataxx.players
import matplotlib.pyplot as plt


def default_eval(board: ataxx.Board):
    black, white, _, _ = board.count()
    if board.turn == ataxx.BLACK:
        return black - white
    else:
        return white - black

class AtaxxEnv(gym.Env):
    def __init__(self, board_dim=7, opponent='alphabeta', depth=2, eval_fn=default_eval, board=None):
        super(AtaxxEnv, self).__init__()

        # Initialize the Ataxx Board
        self.board_dim = board_dim
        self.board = ataxx.Board(board_dim=board_dim) if not board else board
        self.opponent = opponent
        self.depth = depth
        self.eval_fn = eval_fn if eval_fn is not default_eval else default_eval  # default evaluation function

        # Define the action space and observation space
        # Actions: flatten the 7x7 grid to a 49-dimensional space (or any number depending on board_dim)
        self.action_space = spaces.Discrete(board_dim * board_dim)  # Example: 49 actions (7x7 grid)
        self.observation_space = spaces.Box(low=0, high=1, shape=(board_dim, board_dim), dtype=np.float32)

    def sample_action(self, action_index):
        from_row = action_index // (self.board_dim * self.board_dim)
        from_col = (action_index % (self.board_dim * self.board_dim)) // self.board_dim
        to_row = (action_index % self.board_dim) // self.board_dim
        to_col = action_index % self.board_dim
        move = ataxx.Move(from_col, from_row, to_col, to_row)
        return move

    def reset(self):
        """Resets the environment to the initial state."""
        self.board = ataxx.Board(board_dim=self.board_dim)  # Reset the Ataxx board
        return np.array([row[2:-2] for row in self.board._board[2:-2]])  # Return the board state as a numpy array

    def step(self, action):
        """Takes an action and returns the next state, reward, done, and additional info."""
        # Decode the action from the action space (convert index to row, col)
        move = self.sample_action(action)

        done = self.board.gameover()

        # Make the move
        if self.board.is_legal(move):
            self.board.makemove(move)
        elif not done:
            # Invalid action, immediate negative reward
            move = np.random.choice(self.board.legal_moves())
        
        # Get the reward
        if done:
            if self.board.result() == '1-0':  # Example: Player X wins
                reward = 1
            elif self.board.result() == '0-1':  # Example: Player O wins
                reward = -1
            else:
                reward = 0  # Draw
        else:
            reward = 0.1  # Reward some exploration

        # If the game is not over, the opponent takes a turn
        opponent_move = self.get_opponent_move()
        if not done and opponent_move:
            self.board.makemove(opponent_move)

        # Return the new state, reward, done flag, and info
        return np.array([row[2:-2] for row in self.board._board[2:-2]]), reward, done, {}

    def get_opponent_move(self):
        """Get the move for the opponent (random)."""
        return ataxx.players.negamax(self.board, 2)

    def render(self, mode='human'):
        """Render the environment (print the board to the console)."""
        print(self.board)

    def close(self):
        """Clean up the environment (if necessary)."""
        pass

class HierarchicalSoftmaxNet(nn.Module):
    def __init__(self, board_size=7, num_categories=2, num_subcategories=4, num_actions=49):
        super(HierarchicalSoftmaxNet, self).__init__()
        self.fc1 = nn.Linear(board_size * board_size, 256)  # Flatten the board
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        # Hierarchical softmax output layers
        self.level1 = nn.Linear(64, num_categories)  # Categories (e.g., offensive, defensive)
        self.level2 = nn.Linear(64, num_subcategories)  # Sub-categories
        self.level3 = nn.Linear(64, num_actions)  # Final actions (e.g., specific move)

        # Value function head (for value loss)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Output for hierarchical softmax (Level 1: Categories, Level 2: Subcategories, Level 3: Actions)
        level1_probs = torch.softmax(self.level1(x), dim=-1)
        level2_probs = torch.softmax(self.level2(x), dim=-1)
        level3_probs = torch.softmax(self.level3(x), dim=-1)
        
        # Output for value function
        value = self.value_head(x)
        
        return level1_probs, level2_probs, level3_probs, value

def ppo_loss(old_log_probs, log_probs, advantages, values, returns, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
    """
    Compute the PPO loss for the current policy.
    
    Parameters:
    - old_log_probs: The log probabilities from the old policy (before update).
    - log_probs: The log probabilities from the new policy (after update).
    - advantages: The advantage estimates for each state-action pair.
    - values: The state value estimates from the current policy network.
    - returns: The target returns (i.e., the Monte Carlo returns or TD returns).
    - clip_epsilon: The clipping parameter (usually 0.2).
    - value_loss_coef: Coefficient for the value loss.
    - entropy_coef: Coefficient for the entropy term (encourages exploration).
    
    Returns:
    - Total loss, including the policy loss, value loss, and entropy loss.
    """

    # Compute the ratio (r_t)
    ratios = torch.exp(log_probs - old_log_probs)
    
    # Compute the clipped surrogate loss (Policy loss)
    surrogate_loss = torch.min(ratios * advantages, torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages)
    
    # Value loss (Mean Squared Error between value predictions and targets)
    value_loss = (values - returns).pow(2)
    
    # Entropy loss (encourages exploration by penalizing low entropy in the policy distribution)
    entropy_loss = -torch.mean(torch.exp(log_probs) * log_probs)
    
    # Combine the losses
    total_loss = -torch.mean(surrogate_loss) + value_loss_coef * value_loss + entropy_coef * entropy_loss
    
    return total_loss

def compute_gae(rewards, values, next_values, done_flags, gamma=0.99, lam=0.95):
    """
    Compute the Generalized Advantage Estimation (GAE).
    
    Arguments:
    - rewards: List or array of rewards at each time step
    - values: List or array of state values (from the value head) at each time step
    - next_values: List or array of next state values
    - done_flags: List or array of boolean values indicating whether the episode has ended
    - gamma: Discount factor
    - lam: GAE lambda factor
    
    Returns:
    - advantages: Computed advantages at each time step
    - returns: Computed returns (value + advantage)
    """
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    last_gae_lambda = 0
    for t in reversed(range(len(rewards))):
        if done_flags[t]:
            next_value = 0  # No next state if done
        else:
            next_value = next_values[t]
        
        # Compute temporal difference error (delta_t)
        delta = rewards[t] + gamma * next_value - values[t]
        
        # Compute GAE advantage
        advantages[t] = delta + gamma * lam * last_gae_lambda
        returns[t] = advantages[t] + values[t]
        
        # Update the last GAE lambda
        last_gae_lambda = advantages[t]

    return advantages, returns

def train_step(state, action, reward, next_state, done, model, old_log_probs, optimizer, gamma=0.99, lam=0.95):
    """
    Perform one training step (forward pass, backward pass) using PPO with GAE.
    """
    
    # Convert state and next state to tensors
    state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)
    
    # Get the new log probabilities and values from the model
    level1_probs, level2_probs, level3_probs, values = model(state_tensor)
    dist1 = Categorical(level1_probs)
    dist2 = Categorical(level2_probs)
    dist3 = Categorical(level3_probs)
    
    # Sample actions from the distributions
    action1 = dist1.sample()
    action2 = dist2.sample()
    action3 = dist3.sample()
    
    # Combine actions from the hierarchy
    action = action3.item()  # Use the final action from level3
    
    # Compute new log probabilities for PPO update
    new_log_probs = dist1.log_prob(action1) + dist2.log_prob(action2) + dist3.log_prob(action3)
    
    # Collect reward, value, and done information for the current timestep
    rewards = torch.tensor([reward], dtype=torch.float32)
    values = values.detach().squeeze(0)
    
    # Get the next value (could be zero if done)
    next_values = model(torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0))[3].detach().squeeze(0)

    # GAE computation
    advantages, returns = compute_gae(rewards, values, next_values, done_flags=[done], gamma=gamma, lam=lam)
    
    # Compute the advantage (simplified as scalar for now)
    advantage = advantages[0]
    return_value = returns[0]
    
    # Calculate PPO loss
    loss = ppo_loss(old_log_probs, new_log_probs, advantage, values, return_value)
    
    # Backpropagate and update the model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss  # Return the loss for logging purposes

def get_agent(bs=7):
    # Initialize model and optimizer
    model = HierarchicalSoftmaxNet(board_size=bs, num_categories=2, num_subcategories=4, num_actions=bs*bs)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    env = AtaxxEnv()

    # Simulated training loop
    num_episodes = 10
    losses = []
    for episode in tqdm(range(num_episodes)):
        state = env.reset()  # Reset the environment
        done = False
        total_reward = 0
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            
            # Get probabilities from the model
            level1_probs, level2_probs, level3_probs, value = model(state_tensor)
            
            # Select actions using the probabilities from each level
            dist1 = Categorical(level1_probs)
            dist2 = Categorical(level2_probs)
            dist3 = Categorical(level3_probs)
            
            action1 = dist1.sample()
            action2 = dist2.sample()
            action3 = dist3.sample()
            
            # Combine the actions into a single action for the environment
            action = action3.item()
            
            # Store the old log probabilities for PPO update
            old_log_probs = dist1.log_prob(action1) + dist2.log_prob(action2) + dist3.log_prob(action3)
            
            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Train the model using PPO
            loss = train_step(state, action, reward, next_state, done, model, old_log_probs, optimizer)
            
            total_reward += reward
            state = next_state
            losses.append(loss.item())
        print(env.board)
        # print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='PPO Loss', color='blue')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

    return model

def generate_move(model, board, board_size=7):
    """
    Generate a move using a trained hierarchical softmax model.
    
    Parameters:
    - model: The trained PyTorch model.
    - board: The current board.
    - board_size: The size of the board (default is 7 for Ataxx).
    
    Returns:
    - move: The move selected by the model.
    """
    env = AtaxxEnv(board_dim=board_size, board=board)

    # Ensure the board state is a PyTorch tensor and flattened
    board_state =  np.array([row[2:-2] for row in board._board[2:-2]])
    state_tensor = torch.tensor(board_state, dtype=torch.float32).flatten().unsqueeze(0)

    # Get action probabilities from the model
    level1_probs, level2_probs, level3_probs, loss = model(state_tensor)

    # Sample the action from the categorical distributions (one per level)
    dist1 = torch.distributions.Categorical(level1_probs)
    dist2 = torch.distributions.Categorical(level2_probs)
    dist3 = torch.distributions.Categorical(level3_probs)

    action1 = dist1.sample()  # Sample from the first level (category)
    action2 = dist2.sample()  # Sample from the second level (subcategory)
    action3 = dist3.sample()  # Sample from the third level (specific move)

    # Now, we need to combine action2 and action3 to form a valid board position
    # Assuming action2 is a row index, and action3 is a column index, or any other encoding you want to use
    move = env.sample_action(action3.item())
    if not board.is_legal(move):
        move = np.random.choice(board.legal_moves())

    return move
