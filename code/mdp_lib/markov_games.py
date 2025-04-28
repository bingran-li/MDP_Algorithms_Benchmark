import numpy as np
import time


class MarkovGameProcess:
    """
    Base class for Markov Game Processes (MGPs) for two-player zero-sum games.
    
    In a Markov Game Process:
    - We have two players (typically player 0 and player 1)
    - Each state has actions available to the current player
    - Transitions are deterministic or stochastic based on joint actions
    - Rewards are zero-sum (one player's gain is the other's loss)
    """
    def __init__(self, seed=None):
        """
        Initialize MGP solver
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def get_available_actions(self, state, player):
        """
        Get available actions for a player in a given state
        
        Args:
            state: The current game state
            player: The player (0 or 1)
            
        Returns:
            list: Available actions
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def get_next_state(self, state, action, player):
        """
        Get the next state after an action
        
        Args:
            state: The current game state
            action: The action taken
            player: The player taking the action
            
        Returns:
            next_state: The next state
            reward: The reward received
            done: Whether the game is over
            info: Additional information
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def is_terminal(self, state):
        """
        Check if a state is terminal
        
        Args:
            state: The state to check
            
        Returns:
            bool: Whether the state is terminal
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def get_reward(self, state, player):
        """
        Get the reward for a player in a given state
        
        Args:
            state: The game state
            player: The player
            
        Returns:
            float: The reward for the player
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def get_value(self, state, player):
        """
        Get the value of a state for a player (used for evaluation)
        
        Args:
            state: The game state
            player: The player
            
        Returns:
            float: The value for the player
        """
        raise NotImplementedError("Subclass must implement this method")


class TicTacToeGame(MarkovGameProcess):
    """
    Tic-Tac-Toe game implementation as a Markov Game Process
    
    State representation:
    - Board is represented as a 1D list of length k^2 elements
    - 0: Empty, 1: Player 1 (X), 2: Player 2 (O)
    - Player 1 always goes first
    
    Actions:
    - Actions are integers 0 to k^2-1 representing board positions
    - For k=3:
      0 1 2
      3 4 5
      6 7 8
    """
    def __init__(self, board_size=3, seed=None):
        """
        Initialize the Tic-Tac-Toe game
        
        Args:
            board_size (int): Size of the board (k for a k×k board)
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__(seed)
        self.board_size = board_size
        self.board_cells = board_size * board_size
        
        # Generate winning combinations for a k×k board
        self.winning_combinations = self._generate_winning_combinations()
    
    def _generate_winning_combinations(self):
        """
        Generate all possible winning combinations for a k×k board
        
        Returns:
            list: List of winning combinations (each a list of positions)
        """
        k = self.board_size
        combinations = []
        
        # Rows
        for row in range(k):
            combinations.append([row * k + col for col in range(k)])
        
        # Columns
        for col in range(k):
            combinations.append([row * k + col for row in range(k)])
        
        # Diagonals
        combinations.append([i * k + i for i in range(k)])  # Main diagonal
        combinations.append([i * k + (k - 1 - i) for i in range(k)])  # Other diagonal
        
        return combinations
    
    def get_initial_state(self):
        """
        Get the initial (empty) state of the game
        
        Returns:
            tuple: Initial state (empty board, player 0's turn)
        """
        # Initial state: Empty board (0s) and player 0's turn
        return tuple([0] * self.board_cells), 0
    
    def get_available_actions(self, state, player=None):
        """
        Get available actions (empty squares) in a given state
        
        Args:
            state: Tuple (board, current_player)
            player: Not used, we use the current player from state
            
        Returns:
            list: Available actions (indices of empty squares)
        """
        board, _ = state
        return [i for i in range(self.board_cells) if board[i] == 0]
    
    def get_next_state(self, state, action, player=None):
        """
        Get the next state after an action
        
        Args:
            state: Tuple (board, current_player)
            action: Board position (0 to k^2-1)
            player: Not used, we use the current player from state
            
        Returns:
            next_state: The next state
            reward: The reward received
            done: Whether the game is over
            info: Additional information
        """
        board, current_player = state
        
        # Validate action
        if action not in self.get_available_actions(state):
            raise ValueError(f"Invalid action {action} for state {state}")
        
        # Create a new board with the action applied
        new_board = list(board)
        new_board[action] = current_player + 1  # 1 for player 0, 2 for player 1
        new_board = tuple(new_board)
        
        # Switch player (0->1, 1->0)
        next_player = 1 - current_player
        
        # Check if game is over
        done = self.is_terminal((new_board, next_player))
        
        # Calculate reward
        reward = 0
        if done:
            winner = self.get_winner(new_board)
            if winner == 1:  # Player 0 won
                reward = 1 if current_player == 0 else -1
            elif winner == 2:  # Player 1 won
                reward = 1 if current_player == 1 else -1
        
        return (new_board, next_player), reward, done, {}
    
    def is_terminal(self, state):
        """
        Check if a state is terminal (game over)
        
        Args:
            state: Tuple (board, current_player)
            
        Returns:
            bool: Whether the state is terminal
        """
        board, _ = state
        
        # Check for a winner
        if self.get_winner(board) is not None:
            return True
        
        # Check for a draw (no empty squares)
        return 0 not in board
    
    def get_winner(self, board):
        """
        Get the winner of the game
        
        Args:
            board: The game board
            
        Returns:
            int or None: 1 for player 0, 2 for player 1, None for no winner
        """
        for combo in self.winning_combinations:
            if board[combo[0]] != 0:
                if all(board[combo[0]] == board[pos] for pos in combo):
                    return board[combo[0]]  # 1 for player 0, 2 for player 1
        return None
    
    def get_reward(self, state, player):
        """
        Get the reward for a player in a given state
        
        Args:
            state: Tuple (board, current_player)
            player: The player (0 or 1)
            
        Returns:
            float: The reward for the player
        """
        board, _ = state
        winner = self.get_winner(board)
        
        if winner is None:
            return 0
        
        # +1 for winning, -1 for losing, 0 for draw
        player_mark = player + 1  # 1 for player 0, 2 for player 1
        if winner == player_mark:
            return 1
        else:
            return -1
    
    def get_value(self, state, player):
        """
        Get the value of a state for a player
        
        Args:
            state: Tuple (board, current_player)
            player: The player (0 or 1)
            
        Returns:
            float: The value for the player
        """
        if self.is_terminal(state):
            return self.get_reward(state, player)
        return 0  # Intermediate state
    
    def print_board(self, board):
        """
        Print the current board state
        
        Args:
            board: The game board
        """
        k = self.board_size
        symbols = [' ', 'X', 'O']
        
        # Create divider line based on board size
        divider = "-" * (4 * k + 1)
        
        print(divider)
        for row in range(k):
            row_cells = []
            for col in range(k):
                idx = row * k + col
                row_cells.append(symbols[board[idx]])
            print("| " + " | ".join(row_cells) + " |")
            print(divider)
            
    def simulate_random_opponent(self, state, policy, max_steps=100):
        """
        Simulate a game against a random opponent
        
        Args:
            state: Initial state
            policy: Policy for player 0 (function mapping state to action)
            max_steps: Maximum steps to simulate
            
        Returns:
            tuple: (game_state, winner, steps)
        """
        board, current_player = state
        steps = 0
        
        while steps < max_steps and not self.is_terminal((board, current_player)):
            actions = self.get_available_actions((board, current_player))
            
            if not actions:
                break
            
            if current_player == 0:  # Our player
                action = policy((board, current_player))
            else:  # Random opponent
                action = np.random.choice(actions)
            
            (board, current_player), _, done, _ = self.get_next_state(
                (board, current_player), action
            )
            steps += 1
            
            if done:
                break
        
        winner = self.get_winner(board)
        return (board, current_player), winner, steps


class MGPValueIteration:
    """
    Value Iteration algorithm for Markov Game Processes (specifically zero-sum games)
    """
    def __init__(self, game, seed=None):
        """
        Initialize the Value Iteration solver
        
        Args:
            game: The game (MGP) to solve
            seed (int, optional): Random seed for reproducibility
        """
        self.game = game
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Store value function for all states
        self.values = {}
        # Store policies for both players
        self.policies = {0: {}, 1: {}}
    
    def encode_state(self, state):
        """
        Encode a state to a hashable representation
        
        Args:
            state: The state to encode
            
        Returns:
            Hashable state representation
        """
        # For TicTacToe, state is already hashable (tuple)
        return state
    
    def solve(self, max_iterations=1000, tolerance=1e-6):
        """
        Solve the Markov Game using Value Iteration
        
        Args:
            max_iterations (int): Maximum iterations for convergence
            tolerance (float): Convergence tolerance
            
        Returns:
            dict: Value function for player 0 (assumes zero-sum)
        """
        print("Solving game using Value Iteration...")
        start_time = time.time()
        
        # Initialize state space by exploring the game
        print("Initializing state space...")
        self._initialize_state_space()
        print(f"Found {len(self.values)} possible states")
        
        # Value iteration for zero-sum games
        print("Starting value iteration...")
        for iteration in range(max_iterations):
            delta = 0
            
            # Update state values
            for state_key in self.values:
                state = self._decode_state(state_key)
                
                if self.game.is_terminal(state):
                    # Terminal state values don't change
                    continue
                
                old_value = self.values[state_key]
                
                # Determine current player
                _, current_player = state
                
                # Get minimax value
                new_value = self._get_state_value(state, current_player)
                
                # Update value and track largest change
                self.values[state_key] = new_value
                delta = max(delta, abs(new_value - old_value))
            
            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Max delta: {delta:.6f}")
            
            # Check convergence
            if delta < tolerance:
                print(f"Converged after {iteration+1} iterations!")
                break
        
        # Extract optimal policies for both players
        self._extract_policies()
        
        runtime = time.time() - start_time
        print(f"Solution found in {runtime:.2f} seconds")
        
        return {
            'values': self.values,
            'policies': self.policies,
            'iterations': min(iteration+1, max_iterations),
            'converged': delta < tolerance,
            'runtime': runtime
        }
    
    def _initialize_state_space(self):
        """Initialize the state space by exploring all possible game states"""
        # Queue of states to explore
        queue = [self.game.get_initial_state()]
        explored = set()
        
        while queue:
            state = queue.pop(0)
            state_key = self.encode_state(state)
            
            if state_key in explored:
                continue
            
            # Mark as explored
            explored.add(state_key)
            
            # Initialize value function
            if self.game.is_terminal(state):
                # Terminal state has a fixed value
                self.values[state_key] = self.game.get_reward(state, 0)
            else:
                # Non-terminal state starts with zero value
                self.values[state_key] = 0
                
                # Get available actions and add resulting states to the queue
                _, current_player = state
                actions = self.game.get_available_actions(state)
                
                for action in actions:
                    next_state, _, _, _ = self.game.get_next_state(state, action)
                    queue.append(next_state)
    
    def _get_state_value(self, state, player):
        """
        Get the minimax value of a state for a player
        
        Args:
            state: The current state
            player: The current player (0 or 1)
            
        Returns:
            float: The state value
        """
        actions = self.game.get_available_actions(state)

        if not actions:
            # If no actions (shouldn't happen for non-terminal states), return 0
            return 0

        # For player 0 (maximizing), pick the action with highest value
        # For player 1 (minimizing in zero-sum), pick the action with lowest value
        action_values = []

        for action in actions:
            next_state, reward, done, _ = self.game.get_next_state(state, action)
            next_state_key = self.encode_state(next_state)

            if done:
                # If game ends, use the reward
                action_value = reward if player == 0 else -reward
            else:
                # Otherwise use the value of the next state (negated for player 1)
                # In zero-sum games, V(s) for player 1 = -V(s) for player 0
                next_value = self.values.get(next_state_key, 0)
                action_value = next_value if player == 0 else -next_value

            action_values.append(action_value)

        # Player 0 maximizes, player 1 minimizes
        if player == 0:
            return max(action_values)
        else:
            return min(action_values)
    
    def _extract_policies(self):
        """Extract optimal policies for both players from the value function"""
        for state_key in self.values:
            state = self._decode_state(state_key)
            
            if self.game.is_terminal(state):
                # No policy needed for terminal states
                continue
            
            # Extract policy for current player
            _, current_player = state
            actions = self.game.get_available_actions(state)
            
            best_action = None
            best_value = float('-inf') if current_player == 0 else float('inf')
            
            for action in actions:
                next_state, reward, done, _ = self.game.get_next_state(state, action)
                next_state_key = self.encode_state(next_state)
                
                if done:
                    # If game ends, use the reward
                    action_value = reward if current_player == 0 else -reward
                else:
                    # Otherwise use the value of the next state
                    next_value = self.values.get(next_state_key, 0)
                    action_value = next_value if current_player == 0 else -next_value
                
                # Update best action
                if current_player == 0 and action_value > best_value:
                    best_value = action_value
                    best_action = action
                elif current_player == 1 and action_value < best_value:
                    best_value = action_value
                    best_action = action
            
            # Store optimal action in policy
            self.policies[current_player][state_key] = best_action
    
    def _decode_state(self, state_key):
        """
        Decode a state key back to a state
        
        Args:
            state_key: The encoded state key
            
        Returns:
            The decoded state
        """
        # For TicTacToe, state_key is the state itself
        return state_key
    
    def play_optimal_move(self, state, player=None):
        """
        Get the optimal move for a player in a given state
        
        Args:
            state: The current game state
            player: The player (0 or 1), defaults to current player in state
            
        Returns:
            int: The optimal action
        """
        if player is None:
            _, player = state
            
        state_key = self.encode_state(state)
        
        # Check if we have a policy for this state
        if state_key in self.policies[player]:
            return self.policies[player][state_key]
        
        # Fallback: compute the best action on the fly
        actions = self.game.get_available_actions(state)
        
        if not actions:
            return None
        
        best_action = None
        best_value = float('-inf') if player == 0 else float('inf')
        
        for action in actions:
            next_state, reward, done, _ = self.game.get_next_state(state, action)
            next_state_key = self.encode_state(next_state)
            
            if done:
                action_value = reward if player == 0 else -reward
            else:
                next_value = self.values.get(next_state_key, 0)
                action_value = next_value if player == 0 else -next_value
            
            if player == 0 and action_value > best_value:
                best_value = action_value
                best_action = action
            elif player == 1 and action_value < best_value:
                best_value = action_value
                best_action = action
        
        return best_action


def play_tic_tac_toe_game(solver=None, player_human=0):
    """
    Play a Tic-Tac-Toe game against the optimal policy
    
    Args:
        solver: The MGPValueIteration solver with optimal policy
        player_human: Which player the human plays (0 or 1)
    """
    game = TicTacToeGame()
    
    if solver is None:
        # If no solver provided, create one and solve the game
        solver = MGPValueIteration(game)
        print("Solving Tic-Tac-Toe game... (this may take a while)")
        result = solver.solve()
        print(f"Game solved in {result['iterations']} iterations")
    
    state = game.get_initial_state()
    board, current_player = state
    
    print("Welcome to Tic-Tac-Toe!")
    print("You are playing as", "X" if player_human == 0 else "O")
    print("Positions are numbered as follows:")
    print("-------------")
    print("| 0 | 1 | 2 |")
    print("-------------")
    print("| 3 | 4 | 5 |")
    print("-------------")
    print("| 6 | 7 | 8 |")
    print("-------------")
    
    while not game.is_terminal(state):
        board, current_player = state
        
        # Print current board
        print("\nCurrent board:")
        game.print_board(board)
        
        if current_player == player_human:
            # Human's turn
            valid_actions = game.get_available_actions(state)
            
            action = None
            while action not in valid_actions:
                try:
                    action = int(input(f"Enter your move (valid: {valid_actions}): "))
                except ValueError:
                    print("Invalid input, please enter a number")
            
            # Make the move
            state, _, _, _ = game.get_next_state(state, action)
            
        else:
            # AI's turn
            print("AI is thinking...")
            action = solver.play_optimal_move(state)
            print(f"AI plays: {action}")
            
            # Make the move
            state, _, _, _ = game.get_next_state(state, action)
    
    # Final board
    board, _ = state
    print("\nFinal board:")
    game.print_board(board)
    
    # Determine winner
    winner = game.get_winner(board)
    if winner == 1:
        print("X wins!")
    elif winner == 2:
        print("O wins!")
    else:
        print("It's a draw!")


def solve_tic_tac_toe_example():
    """
    Example of solving and analyzing the Tic-Tac-Toe game
    """
    # Create the game and solver
    game = TicTacToeGame(seed=42)
    solver = MGPValueIteration(game, seed=42)
    
    # Solve the game
    result = solver.solve(max_iterations=1000)
    
    # Analyze the initial state value
    initial_state = game.get_initial_state()
    initial_state_key = solver.encode_state(initial_state)
    
    print("\nTic-Tac-Toe Analysis:")
    print(f"Total states: {len(solver.values)}")
    print(f"Value of initial state: {solver.values[initial_state_key]}")
    
    # With optimal play from both sides, Tic-Tac-Toe should be a draw
    # (value close to 0)
    if abs(solver.values[initial_state_key]) < 0.1:
        print("As expected, Tic-Tac-Toe is a draw with optimal play from both sides")
    elif solver.values[initial_state_key] > 0:
        print("Surprisingly, first player (X) can force a win with optimal play")
    else:
        print("Surprisingly, second player (O) can force a win with optimal play")
    
    # Return the solver for potential gameplay
    return solver


if __name__ == "__main__":
    # Solve the game
    solver = solve_tic_tac_toe_example()
    
    # Allow the user to play against the optimal policy
    play_tic_tac_toe_game(solver) 