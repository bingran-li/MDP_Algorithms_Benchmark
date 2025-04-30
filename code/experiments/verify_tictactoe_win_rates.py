#!/usr/bin/env python
"""
Verify win rates of different starting positions in Tic-Tac-Toe against a random player.

This script simulates many games of Tic-Tac-Toe where:
1. The first player (X) makes a specific first move
2. Then follows an optimal policy
3. The second player (O) makes random moves

For each possible starting position (0-8), we record win/draw/loss statistics.
"""

import numpy as np
import time
import sys
import os
from collections import defaultdict
from tabulate import tabulate

# Add the parent directory to the path to use existing modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import TicTacToe implementation from your existing code
from mdp_lib.markov_games import TicTacToeGame, MGPValueIteration

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

def verify_win_rates(num_simulations=10000):
    """
    Verify win rates for each possible first move in 3x3 Tic-Tac-Toe.
    
    Args:
        num_simulations: Number of simulations to run per position
        
    Returns:
        Dictionary with statistics for each starting position
    """
    print(f"Running {num_simulations} simulations per starting position...")
    
    # Create the game and solver
    game = TicTacToeGame(seed=SEED)
    solver = MGPValueIteration(game, seed=SEED)
    
    # Solve the game to get the optimal policy
    print("Solving the game to find the optimal policy...")
    result = solver.solve()
    print(f"Game solved in {result['iterations']} iterations")
    
    # Statistics for each starting position
    position_stats = {}
    
    # Positions for a 3x3 board:
    # 0 | 1 | 2
    # ---------
    # 3 | 4 | 5
    # ---------
    # 6 | 7 | 8
    
    for position in range(9):
        row, col = position // 3, position % 3
        print(f"\nAnalyzing position {position} (row {row}, col {col})...")
        
        wins = draws = losses = 0
        
        # Define a policy that uses the specified first move and then optimal policy
        def first_move_then_optimal(state):
            board, current_player = state
            # If this is the first move (empty board), use the specified position
            if sum(1 for cell in board if cell != 0) == 0:
                return position
            # Otherwise, use the optimal policy from the solver
            return solver.play_optimal_move(state)
        
        # Run simulations
        for i in range(num_simulations):
            # Simulate a game using our policy against a random player
            final_state, winner, steps = game.simulate_random_opponent(
                game.get_initial_state(), first_move_then_optimal)
            
            if winner == 1:  # First player (X) wins
                wins += 1
            elif winner == 2:  # Second player (O) wins
                losses += 1
            else:  # Draw
                draws += 1
        
        # Calculate win rate and store statistics
        win_rate = wins / num_simulations * 100
        draw_rate = draws / num_simulations * 100
        loss_rate = losses / num_simulations * 100
        
        position_stats[position] = {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate
        }
        
        print(f"Position {position}: Win rate = {win_rate:.1f}%, Draw rate = {draw_rate:.1f}%, Loss rate = {loss_rate:.1f}%")
    
    return position_stats

def print_board_with_stats(stats):
    """Print a visual representation of the board with win rates"""
    print("\nWin rates for each starting position:")
    
    # Create a 3x3 grid to visualize the board with win rates
    board = np.zeros((3, 3), dtype=object)
    for pos in range(9):
        row, col = pos // 3, pos % 3
        board[row, col] = f"{stats[pos]['win_rate']:.1f}%"
    
    # Print the board
    print("-" * 25)
    for row in range(3):
        print("| ", end="")
        for col in range(3):
            print(f"{board[row, col]:^5} | ", end="")
        print("\n" + "-" * 25)

def create_comparison_table(stats):
    """Create a table comparing stats for all positions"""
    table_data = []
    
    for pos in range(9):
        row, col = pos // 3, pos % 3
        s = stats[pos]
        
        # Determine position type
        if pos == 4:
            pos_type = "Center"
        elif pos in [0, 2, 6, 8]:
            pos_type = "Corner"
        else:
            pos_type = "Edge"
            
        table_data.append([
            pos,
            f"({row}, {col})",
            pos_type,
            f"{s['win_rate']:.1f}%",
            f"{s['draw_rate']:.1f}%",
            f"{s['loss_rate']:.1f}%",
            s['wins'],
            s['draws'],
            s['losses']
        ])
    
    headers = ["Position", "Coordinates", "Type", "Win %", "Draw %", "Loss %", "Wins", "Draws", "Losses"]
    print("\nDetailed statistics for all starting positions:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    # Number of simulations per position
    num_simulations = 10000
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        try:
            num_simulations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of simulations: {sys.argv[1]}. Using default: {num_simulations}")
    
    start_time = time.time()
    stats = verify_win_rates(num_simulations)
    end_time = time.time()
    
    print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds")
    
    # Print board visualization with win rates
    print_board_with_stats(stats)
    
    # Create comparison table
    create_comparison_table(stats)
    
    # Determine best starting position
    best_pos = max(stats.keys(), key=lambda pos: stats[pos]['win_rate'])
    row, col = best_pos // 3, best_pos % 3
    win_rate = stats[best_pos]['win_rate']
    
    print(f"\nBest starting position: {best_pos} (row {row}, col {col}) with {win_rate:.1f}% win rate")
    
    # Explain theoretical vs empirical results
    print("\nExplanation of results:")
    print("1. Against a purely random player, some positions give better win rates than others")
    print("2. The Nash equilibrium strategy (optimal against perfect play) may not maximize")
    print("   win rate against a random player because:")
    print("   - It plays defensively to guarantee the best worst-case outcome")
    print("   - It assumes the opponent will make optimal counter-moves")
    print("   - Random play can sometimes be beneficial by making unexpected moves")
    print("3. Theoretical optimal play is designed for perfect opponents, not random ones")

if __name__ == "__main__":
    main()