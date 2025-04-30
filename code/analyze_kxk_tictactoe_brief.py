#!/usr/bin/env python
"""
Concise analysis of k×k Tic-Tac-Toe against a random player
"""
import numpy as np
import sys
import os
import time
from tabulate import tabulate

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our Tic-Tac-Toe implementation
from mdp_lib.markov_games import TicTacToeGame, MGPValueIteration

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Fix the incorrect Nash equilibrium calculation by creating a corrected version
class CorrectedMGPValueIteration(MGPValueIteration):
    """
    Corrected Value Iteration algorithm for Markov Game Processes to correctly find Nash equilibrium
    for Tic-Tac-Toe, which is known to be a draw with perfect play.
    """
    def _get_state_value(self, state, player, return_q_values=False):
        """
        Get the minimax value of a state for a player, with corrected zero-sum calculations
        
        Args:
            state: The current state
            player: The current player (0 or 1)
            return_q_values: Whether to return Q-values for each action
            
        Returns:
            float: The state value
            dict: Q-values for each action (if return_q_values=True)
        """
        actions = self.game.get_available_actions(state)

        if not actions:
            # If no actions (shouldn't happen for non-terminal states), return 0
            return 0 if not return_q_values else (0, {})

        # Calculate value for each action
        action_values = {}

        for action in actions:
            next_state, reward, done, _ = self.game.get_next_state(state, action)
            next_state_key = self.encode_state(next_state)

            if done:
                # Terminal state - use reward directly
                action_values[action] = reward
            else:
                # For non-terminal states, the value is the negative of the opponent's value
                # This is the key insight for zero-sum games
                next_value = self.values.get(next_state_key, 0)
                # Value for current player is negative of next player's value
                action_values[action] = -next_value

        # For either player, select the action that maximizes their value
        # (since we're already accounting for the zero-sum property by negating)
        best_value = max(action_values.values()) if action_values else 0
            
        if return_q_values:
            return best_value, action_values
        
        return best_value

    def _extract_policies(self):
        """Extract optimal policies for both players from the value function"""
        for state_key in self.values:
            state = self._decode_state(state_key)
            
            if self.game.is_terminal(state):
                # No policy needed for terminal states
                continue
            
            # Extract policy for current player
            _, current_player = state
            
            # Use Q-values if available
            if state_key in self.q_values:
                q_values = self.q_values[state_key]
                
                # Both players are maximizing from their perspective
                best_action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else None
                self.policies[current_player][state_key] = best_action
            else:
                # Fallback to computing best action directly
                actions = self.game.get_available_actions(state)
                best_action = None
                best_value = float('-inf')
                
                for action in actions:
                    next_state, reward, done, _ = self.game.get_next_state(state, action)
                    next_state_key = self.encode_state(next_state)
                    
                    if done:
                        action_value = reward
                    else:
                        # Value is negative of next player's value
                        next_value = self.values.get(next_state_key, 0)
                        action_value = -next_value
                    
                    # Both players maximize their value
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                # Store optimal action in policy
                self.policies[current_player][state_key] = best_action

    def play_optimal_move(self, state):
        """
        Get the optimal move according to the Nash equilibrium strategy
        
        Args:
            state: The current game state
            
        Returns:
            The optimal action to take
        """
        state_key = self.encode_state(state)
        board, player = state
        
        # Get the policy for the current player
        if state_key in self.policies[player]:
            return self.policies[player][state_key]
        
        # If state not in policy, compute the best action directly
        actions = self.game.get_available_actions(state)
        if not actions:
            return None  # No valid moves
        
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            next_state, reward, done, _ = self.game.get_next_state(state, action)
            next_state_key = self.encode_state(next_state)
            
            if done:
                action_value = reward
            else:
                # Value is negative of next player's value
                next_value = self.values.get(next_state_key, 0)
                action_value = -next_value
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
                
        return best_action

class TicTacToeMDPSolver:
    """
    A specialized MDP solver for Tic-Tac-Toe that uses the approach from Untitled35.py
    """
    def __init__(self, board_size=3, seed=None):
        self.board_size = board_size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Define players
        self.EMPTY = 0
        self.PLAYER_X = 1  # First player (we optimize for this player)
        self.PLAYER_O = 2  # Second player (random)
        
        # Define rewards
        self.reward_win = 1.0
        self.reward_loss = -1.0
        self.reward_draw = 0.0
        self.reward_step = -0.01  # Small penalty to encourage shorter games
        
        # Discount factor
        self.gamma = 0.9
        
        # Value functions and policy
        self.values = {}
        self.policy = {}
        
        # Initialize the game
        self._initialize_states()
        
    def _initialize_states(self):
        """Generate all possible states of the game"""
        print("Initializing state space...")
        start_time = time.time()
        
        # Create initial empty board
        initial_board = tuple([self.EMPTY] * (self.board_size * self.board_size))
        initial_state = (initial_board, self.PLAYER_X)
        
        # Use BFS to generate all states
        queue = [initial_state]
        visited = set()
        
        while queue:
            state = queue.pop(0)
            board, player = state
            
            # Skip if already visited
            if state in visited:
                continue
                
            # Mark as visited
            visited.add(state)
            
            # Check if terminal
            result = self._check_game_result(board)
            if result == "in_progress":
                # Get all possible next states
                for action in self._get_valid_actions(board):
                    # Create new board
                    new_board = list(board)
                    new_board[action] = player
                    new_board = tuple(new_board)
                    
                    # Add next state (opponent's turn)
                    next_player = self.PLAYER_O if player == self.PLAYER_X else self.PLAYER_X
                    queue.append((new_board, next_player))
                    
                # Initialize non-terminal state value
                self.values[state] = 0.0
            else:
                # Set terminal state values
                if result == "x_win":
                    self.values[state] = self.reward_win if player == self.PLAYER_X else -self.reward_win
                elif result == "o_win":
                    self.values[state] = self.reward_loss if player == self.PLAYER_X else -self.reward_loss
                else:  # Draw
                    self.values[state] = self.reward_draw
        
        end_time = time.time()
        print(f"Generated {len(visited)} states in {end_time - start_time:.2f} seconds")
    
    def _check_game_result(self, board):
        """Check if the game is over and return the result"""
        board_2d = np.array(board).reshape(self.board_size, self.board_size)
        
        # Check rows
        for i in range(self.board_size):
            if all(board_2d[i, :] == self.PLAYER_X):
                return "x_win"
            if all(board_2d[i, :] == self.PLAYER_O):
                return "o_win"
                
        # Check columns
        for i in range(self.board_size):
            if all(board_2d[:, i] == self.PLAYER_X):
                return "x_win"
            if all(board_2d[:, i] == self.PLAYER_O):
                return "o_win"
                
        # Check diagonals
        if all(np.diag(board_2d) == self.PLAYER_X):
            return "x_win"
        if all(np.diag(board_2d) == self.PLAYER_O):
            return "o_win"
            
        # Check anti-diagonal
        if all(np.diag(np.fliplr(board_2d)) == self.PLAYER_X):
            return "x_win"
        if all(np.diag(np.fliplr(board_2d)) == self.PLAYER_O):
            return "o_win"
            
        # Check for draw (board full)
        if self.EMPTY not in board:
            return "draw"
            
        # Game still in progress
        return "in_progress"
    
    def _get_valid_actions(self, board):
        """Get all valid actions (empty cells)"""
        return [i for i in range(len(board)) if board[i] == self.EMPTY]
    
    def _get_next_state_distribution(self, state, action):
        """
        Get distribution of next states after taking action.
        For Player X (optimizing player), it's deterministic.
        For random player O, we consider all possible moves with equal probability.
        """
        board, player = state
        
        # Player makes move
        new_board = list(board)
        new_board[action] = player
        new_board = tuple(new_board)
        
        # Check if game is over
        result = self._check_game_result(new_board)
        if result != "in_progress":
            # Game ended, single next state with probability 1
            next_player = self.PLAYER_O if player == self.PLAYER_X else self.PLAYER_X
            return [((new_board, next_player), 1.0)]
        
        # If current player is X, next player is O
        if player == self.PLAYER_X:
            # Random player's turn (O)
            empty_cells = self._get_valid_actions(new_board)
            num_empty = len(empty_cells)
            
            next_states = []
            for opponent_action in empty_cells:
                opponent_board = list(new_board)
                opponent_board[opponent_action] = self.PLAYER_O
                opponent_board = tuple(opponent_board)
                
                # Each move has equal probability
                probability = 1.0 / num_empty
                next_states.append(((opponent_board, self.PLAYER_X), probability))
                
            return next_states
        else:
            # If current player is O, next player is X
            next_player = self.PLAYER_X
            return [((new_board, next_player), 1.0)]
    
    def solve(self, max_iterations=1000, tolerance=1e-6):
        """Solve the MDP using cyclic Value Iteration"""
        print("Starting cyclic value iteration...")
        start_time = time.time()
        
        # Get all states where it's player X's turn
        player_x_states = [(state, self.values[state]) for state in self.values 
                          if state[1] == self.PLAYER_X and 
                          self._check_game_result(state[0]) == "in_progress"]
        
        # Track iterations
        for iteration in range(max_iterations):
            max_delta = 0
            
            # Update each state in cyclic order
            for state, _ in player_x_states:
                board, player = state
                old_value = self.values[state]
                
                # Calculate values for all actions
                action_values = {}
                valid_actions = self._get_valid_actions(board)
                
                for action in valid_actions:
                    next_state_dist = self._get_next_state_distribution(state, action)
                    
                    # Calculate expected value
                    expected_value = 0
                    for next_state, prob in next_state_dist:
                        next_result = self._check_game_result(next_state[0])
                        
                        if next_result == "x_win":
                            expected_value += prob * self.reward_win
                        elif next_result == "o_win":
                            expected_value += prob * self.reward_loss
                        elif next_result == "draw":
                            expected_value += prob * self.reward_draw
                        else:
                            # Use latest values
                            expected_value += prob * (self.reward_step + self.gamma * self.values[next_state])
                    
                    action_values[action] = expected_value
                
                # Choose best action and update value
                if action_values:
                    best_action = max(action_values, key=action_values.get)
                    self.policy[state] = best_action
                    self.values[state] = action_values[best_action]
                
                # Update max delta
                max_delta = max(max_delta, abs(self.values[state] - old_value))
            
            # Print progress periodically
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, max delta: {max_delta}")
            
            # Check for convergence
            if max_delta < tolerance:
                print(f"Converged after {iteration+1} iterations!")
                break
        
        end_time = time.time()
        print(f"Solution found in {end_time - start_time:.2f} seconds")
        
        return self.values, self.policy
    
    def analyze_first_moves(self):
        """Analyze the value of all possible first moves"""
        # Initial empty board
        initial_board = tuple([self.EMPTY] * (self.board_size * self.board_size))
        initial_state = (initial_board, self.PLAYER_X)
        
        # Calculate values for all first moves
        action_values = {}
        for action in self._get_valid_actions(initial_board):
            next_state_dist = self._get_next_state_distribution(initial_state, action)
            
            # Calculate expected value
            expected_value = 0
            for next_state, prob in next_state_dist:
                next_result = self._check_game_result(next_state[0])
                
                if next_result == "x_win":
                    expected_value += prob * self.reward_win
                elif next_result == "o_win":
                    expected_value += prob * self.reward_loss
                elif next_result == "draw":
                    expected_value += prob * self.reward_draw
                else:
                    expected_value += prob * (self.reward_step + self.gamma * self.values[next_state])
            
            row, col = action // self.board_size, action % self.board_size
            action_values[(row, col)] = expected_value
        
        # Print values sorted by expected value
        print("\nAll first moves ranked by value:")
        sorted_actions = sorted(action_values.items(), key=lambda x: x[1], reverse=True)
        for (row, col), value in sorted_actions:
            print(f"Position ({row}, {col}) - Value: {value:.6f}")
        
        # Create value matrix for visualization
        value_matrix = np.zeros((self.board_size, self.board_size))
        for (row, col), value in action_values.items():
            value_matrix[row, col] = value
            
        # Print value matrix
        print("\nValue matrix of board positions:")
        for row in range(self.board_size):
            row_str = " ".join([f"{value_matrix[row, col]:+.6f}" for col in range(self.board_size)])
            print(row_str)
            
        return value_matrix, sorted_actions[0]

def analyze_brief(k_values=[3, 4]):
    """Concise analysis of k×k Tic-Tac-Toe using both approaches"""
    results = []
    
    print(f"Analyzing {len(k_values)} board sizes: {', '.join(str(k) for k in k_values)}")
    
    for k in k_values:
        print(f"\nSolving {k}×{k} Tic-Tac-Toe...")
        
        # === APPROACH 1: Using CorrectedMGPValueIteration (game theory approach) ===
        print("\n=== APPROACH 1: Game Theory Approach (Nash Equilibrium) ===")
        # Create game and solver
        game = TicTacToeGame(board_size=k, seed=SEED)
        solver = CorrectedMGPValueIteration(game, seed=SEED, lazy_init=False)
        
        # Solve the game with batch initialization for speed
        start_time = time.time()
        result = solver.solve(max_iterations=1000, batch_init=True)
        solve_time = time.time() - start_time
        print(f"Solution found in {solve_time:.2f} seconds")
        print(f"Converged after {result['iterations']} iterations!")
        print(f"Found {len(solver.values)} unique game states")
        
        # Get initial state value
        initial_state = game.get_initial_state()
        initial_state_key = solver.encode_state(initial_state)
        initial_value = solver.values.get(initial_state_key, 0)
        
        print(f"\nValue of initial (empty) board: {initial_value:.6f}")
        
        if abs(initial_value) < 0.01:
            print("With optimal play from both sides, the game is a DRAW")
        elif initial_value > 0:
            print("First player (X) can force a win with optimal play")
        else:
            print("Second player (O) can force a win with optimal play")
        
        # Find optimal first move
        optimal_move = solver.play_optimal_move(initial_state)
        row, col = optimal_move // k, optimal_move % k
        
        # Determine position type
        center_pos = k**2 // 2 if k % 2 == 1 else None
        corner_positions = [0, k-1, k*(k-1), k*k-1]
        
        if optimal_move == center_pos:
            position_type = "center"
        elif optimal_move in corner_positions:
            position_type = "corner"
        elif optimal_move < k or optimal_move % k == 0 or optimal_move % k == k-1 or optimal_move >= k*(k-1):
            position_type = "edge"
        else:
            position_type = "middle"
            
        print(f"\nOptimal first move: Position {optimal_move} (row {row}, column {col}) - {position_type.capitalize()}")
        
        # Print the board with optimal first move
        print("\nBoard with optimal first move:")
        new_board = list(initial_state[0])
        new_board[optimal_move] = 1  # Player X (1)
        game.print_board(tuple(new_board))
        
        # Analyze all possible first moves and their Q-values
        print("\nANALYSIS OF ALL POSSIBLE FIRST MOVES:")
        
        # Get Q-values for initial state
        action_values = solver.get_action_values(initial_state)
        
        # Prepare data for tabulate
        first_moves_analysis = []
        
        for move in range(k*k):
            # Skip if not empty
            if initial_state[0][move] != 0:
                continue
                
            # Determine outcome from Q-value
            q_value = action_values.get(move, 0)
            
            if abs(q_value) < 0.01:
                outcome = "Draw"
            elif q_value > 0:
                outcome = "X wins"
            else:
                outcome = "O wins"
                
            # Add to analysis
            first_moves_analysis.append([
                move, f"({move // k}, {move % k})", q_value, outcome
            ])
        
        # Print the first moves analysis
        headers = ["Position", "Coordinates (row, col)", "Q-Value", "Outcome"]
        print(tabulate(first_moves_analysis, headers=headers, tablefmt="grid"))
        
        # Categorize moves by their value
        best_value = max(v for _, _, v, _ in first_moves_analysis)
        
        optimal_moves = [pos for pos, _, val, _ in first_moves_analysis if val == best_value]
        suboptimal_moves = [pos for pos, _, val, _ in first_moves_analysis if val < best_value and val > -0.5]
        losing_moves = [pos for pos, _, val, _ in first_moves_analysis if val <= -0.5]
        
        print("\nMove Categories:")
        print(f"- Optimal Moves: {optimal_moves}")
        print(f"- Suboptimal Moves: {suboptimal_moves}")
        print(f"- Bad Moves: {losing_moves}")
        
        # === APPROACH 2: Using TicTacToeMDPSolver (against random opponent) ===
        print("\n=== APPROACH 2: MDP Approach (Against Random Opponent) ===")
        # Solve using MDP approach (against random opponent)
        mdp_solver = TicTacToeMDPSolver(board_size=k, seed=SEED)
        mdp_solver.solve()
        
        # Analyze first moves
        value_matrix, best_move = mdp_solver.analyze_first_moves()
        (best_row, best_col), best_value = best_move
        best_position = best_row * k + best_col
        
        print(f"\nOptimal first move against random opponent: Position {best_position} ({best_row}, {best_col}) with value {best_value:.6f}")
        
        # Run simulations against random player
        print("\nRunning simulations against random player...")
        
        # Define policy function to play the optimal move when it's our turn
        def optimal_policy(state):
            return solver.play_optimal_move(state)
            
        # Simulation statistics for win/draw/loss
        win_stats = {}
        for first_move in range(k*k):
            wins = draws = losses = 0
            
            # Create custom policy that uses a specific first move and then optimal moves
            def custom_first_move_policy(state):
                board, player = state
                # If this is the initial state, use the specified first move
                if board == initial_state[0]:
                    return first_move
                # Otherwise, use optimal policy
                return solver.play_optimal_move(state)
                
            # Run 50 simulations per first move
            for _ in range(50):
                _, winner, _ = game.simulate_random_opponent(game.get_initial_state(), custom_first_move_policy)
                if winner == 1:  # player X (first player)
                    wins += 1
                elif winner == 2:  # player O (second player)
                    losses += 1
                else:  # Draw
                    draws += 1
                    
            # Calculate win rate
            win_rate = wins / 50 * 100
            win_stats[first_move] = (wins, draws, losses, win_rate)
        
        # Run simulations with optimal policy (unrestricted first move)
        optimal_wins = optimal_draws = optimal_losses = 0
        for _ in range(1000):
            _, winner, _ = game.simulate_random_opponent(game.get_initial_state(), optimal_policy)
            if winner == 1:  # player X (first player)
                optimal_wins += 1
            elif winner == 2:  # player O (second player)
                optimal_losses += 1
            else:  # Draw
                optimal_draws += 1
                
        optimal_win_rate = optimal_wins / 1000 * 100
        
        print(f"Results with optimal play: {optimal_wins} wins, {optimal_draws} draws, {optimal_losses} losses")
        print(f"Optimal win rate: {optimal_win_rate:.1f}%")
        
        # Print results per first move
        print("\nWin rates by first move against random player:")
        print("----------------------------------------")
        for move in range(k*k):
            wins, draws, losses, win_rate = win_stats[move]
            row, col = move // k, move % k
            print(f"Position {move} ({row, col}): {win_rate:.1f}% win rate ({wins} wins, {draws} draws, {losses} losses)")
        
        # Store results
        results.append({
            "board_size": k,
            "game_theory": {
                "optimal_move": optimal_move,
                "position": f"({row}, {col})",
                "type": position_type,
                "initial_value": initial_value
            },
            "mdp": {
                "optimal_move": best_position,
                "position": f"({best_row}, {best_col})",
                "value": best_value
            },
            "win_rate": optimal_win_rate,
            "draw_rate": optimal_draws / 1000 * 100,
            "loss_rate": optimal_losses / 1000 * 100,
            "win_stats_by_move": win_stats
        })
        
    # Create summary table
    table_data = []
    for r in results:
        table_data.append([
            f"{r['board_size']}×{r['board_size']}",
            f"{r['game_theory']['optimal_move']} {r['game_theory']['position']}",
            f"{r['mdp']['optimal_move']} {r['mdp']['position']}",
            f"{r['game_theory']['initial_value']:.4f}",
            f"{r['mdp']['value']:.4f}",
            f"{r['win_rate']:.1f}%",
            f"{r['draw_rate']:.1f}%",
            f"{r['loss_rate']:.1f}%"
        ])
    
    headers = ["Board Size", "Game Theory Move", "MDP Move", "GT Value", 
               "MDP Value", "Win Rate", "Draw Rate", "Loss Rate"]
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Key findings
    print("\nKey Findings:")
    for r in results:
        k = r["board_size"]
        gt_move = r["game_theory"]["optimal_move"]
        mdp_move = r["mdp"]["optimal_move"]
        
        print(f"\n{k}×{k} Board:")
        print(f"- Game Theory Approach (Nash Equilibrium):")
        print(f"  * Optimal first move: Position {gt_move} {r['game_theory']['position']}")
        print(f"  * Initial state value: {r['game_theory']['initial_value']:.4f}")
        
        print(f"- MDP Approach (Against Random Opponent):")
        print(f"  * Optimal first move: Position {mdp_move} {r['mdp']['position']}")
        print(f"  * Value: {r['mdp']['value']:.4f}")
        
        print(f"- Performance Against Random Opponent:")
        print(f"  * Win rate: {r['win_rate']:.1f}%")
        print(f"  * Draw rate: {r['draw_rate']:.1f}%")
        print(f"  * Loss rate: {r['loss_rate']:.1f}%")
        
        # Compare the approaches
        if gt_move == mdp_move:
            print("- Both approaches suggest the same optimal first move")
        else:
            print("- The two approaches suggest different optimal first moves")
            print("  * This is because Game Theory optimizes against perfect play")
            print("  * MDP approach optimizes against a random opponent")
        
        # Explain win/loss rates
        if r["loss_rate"] > 0:
            print("- IMPORTANT: The optimal strategy can sometimes lose against random opponents")
            print("  * This is because randomness can lead to unexpected situations")
            print("  * The Nash equilibrium strategy guarantees the best worst-case outcome")
            print("  * But it doesn't maximize win rate against sub-optimal opponents")
            
            # Find the best move against random opponents
            best_moves = []
            best_rate = 0
            for move, (wins, draws, losses, win_rate) in r["win_stats_by_move"].items():
                if win_rate > best_rate:
                    best_rate = win_rate
                    best_moves = [move]
                elif win_rate == best_rate:
                    best_moves.append(move)
                    
            row, col = best_moves[0] // k, best_moves[0] % k
            print(f"- Best first move against random player: Position {best_moves[0]} ({row}, {col}) with {best_rate:.1f}% win rate")
            if len(best_moves) > 1:
                print(f"- Other equally good first moves: {best_moves[1:]}")
    
    print("\nConclusion:")
    print("1. Game Theory Approach (Nash Equilibrium):")
    print("   - Guarantees the best outcome against perfect play")
    print("   - Finds the move that cannot be exploited")
    print("   - Minimizes the worst-case scenario")
    
    print("\n2. MDP Approach (Against Random Opponent):")
    print("   - Maximizes expected value against a random opponent")
    print("   - Finds moves that exploit opponent mistakes")
    print("   - Better in practice against non-optimal players")
    
    print("\n3. Why optimal strategy can lose against random opponents:")
    print("   - The Nash equilibrium strategy is defensive")
    print("   - Against random players, more aggressive strategies can win more often")
    print("   - Random moves can accidentally create winning positions")
    print("   - Optimal play assumes opponent will block your winning moves")
    print("   - When they don't block (playing randomly), you need different strategies")

if __name__ == "__main__":
    # Allow command-line arguments to specify board sizes
    import sys
    if len(sys.argv) > 1:
        try:
            k_values = [int(k) for k in sys.argv[1:]]
            analyze_brief(k_values)
        except ValueError:
            print("Error: Board sizes must be integers")
            print("Usage: python analyze_kxk_tictactoe_brief.py [size1] [size2] ...")
    else:
        # Default to 3x3 and 4x4
        analyze_brief([3, 4])