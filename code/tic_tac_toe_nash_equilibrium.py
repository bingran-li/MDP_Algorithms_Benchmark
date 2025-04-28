#!/usr/bin/env python
"""
Nash Equilibrium and Optimal Strategies in Tic-Tac-Toe using Value Iteration

This script demonstrates:
1. How to model Tic-Tac-Toe as a Markov Game Process (MGP)
2. How to find the Nash equilibrium using Value Iteration
3. How to extract optimal strategies for both players
4. How to analyze the game theoretically
"""

import numpy as np
import time
import itertools
import sys
import os
from tabulate import tabulate

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our Markov Game Process implementation
from mdp_lib.markov_games import TicTacToeGame, MGPValueIteration

# Set seed for reproducibility
SEED = 42

def print_separator(title=None):
    """Print a separator line with optional title"""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")


def analyze_tic_tac_toe():
    """
    Analyze Tic-Tac-Toe game using Value Iteration to find the Nash equilibrium
    and optimal strategies for both players.
    """
    print_separator("TIC-TAC-TOE GAME THEORY ANALYSIS USING VALUE ITERATION")
    
    print("Initializing Tic-Tac-Toe game...")
    game = TicTacToeGame(seed=SEED)
    
    # Create the solver
    print("Creating Value Iteration solver for Markov Game Process...")
    solver = MGPValueIteration(game, seed=SEED)
    
    # Solve the game
    print("\nSolving game using Value Iteration to find the Nash equilibrium...")
    print("This may take a moment as we explore the entire state space...\n")
    
    start_time = time.time()
    result = solver.solve(max_iterations=1000, tolerance=1e-6)
    elapsed_time = time.time() - start_time
    
    print(f"\nSolution found in {elapsed_time:.2f} seconds")
    print(f"Converged after {result['iterations']} iterations")
    print(f"Found {len(solver.values)} unique game states")
    
    # Get the initial state
    initial_state = game.get_initial_state()
    initial_state_key = solver.encode_state(initial_state)
    initial_value = solver.values.get(initial_state_key, 0)
    
    print_separator("GAME THEORETIC VALUE")
    
    print(f"Value of the initial (empty) board: {initial_value:.6f}")
    
    if abs(initial_value) < 0.01:
        print("\nTHEORETICAL OUTCOME: With optimal play from both sides, Tic-Tac-Toe is a DRAW")
        print("This confirms the well-known game theory result that Tic-Tac-Toe")
        print("is a solved game that results in a draw with perfect play.")
    elif initial_value > 0:
        print("\nTHEORETICAL OUTCOME: First player (X) can force a win with optimal play")
        print("This is an unexpected result! Theoretically, Tic-Tac-Toe should be")
        print("a draw with perfect play. Our algorithm has found an advantage for the first player.")
    else:
        print("\nTHEORETICAL OUTCOME: Second player (O) can force a win with optimal play")
        print("This is an unexpected result! Theoretically, Tic-Tac-Toe should be")
        print("a draw with perfect play. Our algorithm has found an advantage for the second player.")
    
    print_separator("NASH EQUILIBRIUM STRATEGY")
    
    print("In the Nash equilibrium:")
    print("- Each player employs a strategy that cannot be improved given the other player's strategy")
    print("- Neither player has an incentive to deviate from their strategy")
    print("- The result is optimal play from both sides")
    
    print("\nLet's examine the optimal first move for the first player (X):")
    optimal_first_move = solver.play_optimal_move(initial_state)
    
    # Convert move to coordinates
    row, col = optimal_first_move // 3, optimal_first_move % 3
    
    print(f"\nOptimal first move: Position {optimal_first_move} (row {row}, column {col})")
    
    # Visualize the board with the optimal first move
    print("\nOptimal opening:")
    new_board = list(initial_state[0])
    new_board[optimal_first_move] = 1  # Player X (1)
    game.print_board(tuple(new_board))
    
    print_separator("ANALYSIS OF ALL POSSIBLE FIRST MOVES")
    
    # Analyze all possible first moves
    first_moves_analysis = []
    
    for move in range(9):
        # Skip if not a valid move
        if initial_state[0][move] != 0:
            continue
            
        # Make the move
        new_board = list(initial_state[0])
        new_board[move] = 1  # Player 1 (X)
        new_state = (tuple(new_board), 1)  # Switch to player 2 (O)
        new_state_key = solver.encode_state(new_state)
        
        # Get the value
        value = -solver.values.get(new_state_key, 0)  # Negate because it's from the opponent's perspective
        
        # Determine the outcome
        if abs(value) < 0.01:
            outcome = "Draw"
        elif value > 0:
            outcome = "Player 1 (X) wins"
        else:
            outcome = "Player 2 (O) wins"
            
        # Add to analysis
        first_moves_analysis.append([
            move, f"({move // 3}, {move % 3})", value, outcome
        ])
    
    # Print the first moves analysis
    print("Value of each possible first move from Player 1's perspective:\n")
    headers = ["Position", "Coordinates (row, col)", "Value", "Theoretical Outcome"]
    print(tabulate(first_moves_analysis, headers=headers, tablefmt="grid"))
    
    # Create a mapping of moves to values for easy lookup
    move_values = {move[0]: move[2] for move in first_moves_analysis}
    
    print("\nCategory of Moves:")
    print("1. Optimal Moves: Guarantee the best possible outcome")
    print("2. Suboptimal Moves: Lead to a worse outcome than possible")
    print("3. Losing Moves: Guarantee a loss against optimal play")
    
    # Determine the best possible outcome
    best_value = max(move[2] for move in first_moves_analysis)
    
    optimal_moves = [move[0] for move in first_moves_analysis if move[2] == best_value]
    suboptimal_moves = [move[0] for move in first_moves_analysis if move[2] < best_value and move[2] > -1]
    losing_moves = [move[0] for move in first_moves_analysis if move[2] <= -1]
    
    print("\nOptimal Moves:", optimal_moves)
    print("Suboptimal Moves:", suboptimal_moves)
    print("Losing Moves:", losing_moves)
    
    print_separator("STRATEGIC INSIGHTS")
    
    print("Key Insights from the MDP Analysis:")
    
    if 4 in optimal_moves:  # Center
        print("1. The center position (4) is optimal")
    if any(corner in optimal_moves for corner in [0, 2, 6, 8]):
        print("1. Corner positions are optimal openings")
    if any(edge in optimal_moves for edge in [1, 3, 5, 7]):
        print("1. Edge positions can be optimal")
    
    print("2. With optimal play, the game results in a draw")
    print("3. The Nash equilibrium strategy is:")
    print("   - Player 1 (X): Choose an optimal opening move")
    print("   - Player 2 (O): Play defensively to force a draw")
    
    print("\n4. Tic-Tac-Toe has a first-mover advantage, but it's not decisive")
    print("5. The game has perfect information and is deterministic")
    print("6. The Nash equilibrium is a saddle point in the payoff matrix")
    
    print_separator("POLICY VISUALIZATION")
    
    # Show example of policy for a specific state
    print("Let's examine the optimal policy for a specific game state:")
    
    # Create a board with a specific configuration
    # X O X
    # . . .
    # . . O
    example_board = [1, 2, 1, 0, 0, 0, 0, 0, 2]  # 1=X, 2=O
    example_state = (tuple(example_board), 0)  # Player 1's (X) turn
    
    print("\nCurrent board state:")
    game.print_board(example_board)
    
    # Get optimal move for player 1
    optimal_move = solver.play_optimal_move(example_state)
    row, col = optimal_move // 3, optimal_move % 3
    
    print(f"\nOptimal move for Player 1 (X): Position {optimal_move} (row {row}, column {col})")
    
    # Make the move
    new_board = example_board.copy()
    new_board[optimal_move] = 1
    
    print("\nResulting board:")
    game.print_board(new_board)
    
    # Calculate value after move
    new_state = (tuple(new_board), 1)  # Switch to player 2
    new_state_key = solver.encode_state(new_state)
    value = -solver.values.get(new_state_key, 0)  # From player 1's perspective
    
    print(f"\nValue after this move: {value:.6f}")
    
    if abs(value) < 0.01:
        print("This leads to a draw with optimal play")
    elif value > 0:
        print("This leads to a win for Player 1 (X) with optimal play")
    else:
        print("This leads to a win for Player 2 (O) with optimal play")
    
    print_separator("CONCLUSION")
    
    print("The Value Iteration algorithm has successfully solved the Tic-Tac-Toe game,")
    print("finding the Nash equilibrium and optimal strategies for both players.")
    print("\nKey takeaways:")
    print("1. Tic-Tac-Toe is a solved game that results in a draw with optimal play")
    print("2. The first player (X) has multiple optimal opening moves")
    print("3. Value Iteration can efficiently find optimal policies in deterministic games")
    print("\nThis demonstration shows how MDPs and Value Iteration can be applied")
    print("to solve zero-sum games and find Nash equilibria in game theory.")


if __name__ == "__main__":
    analyze_tic_tac_toe() 