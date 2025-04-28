#!/usr/bin/env python
"""
Simple analysis of k×k Tic-Tac-Toe against a random player
"""
import numpy as np
import time
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our Tic-Tac-Toe implementation
from mdp_lib.markov_games import TicTacToeGame, MGPValueIteration

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

def analyze_tictactoe(k_values=[3, 4]):
    """Simple analysis of k×k Tic-Tac-Toe"""
    
    print(f"Analyzing {len(k_values)} board sizes: {', '.join(str(k) for k in k_values)}")
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"BOARD SIZE {k}×{k}")
        print(f"{'='*50}")
        
        # Create game and solver
        start_time = time.time()
        print(f"Creating {k}×{k} Tic-Tac-Toe game...")
        game = TicTacToeGame(board_size=k, seed=SEED)
        solver = MGPValueIteration(game, seed=SEED)
        
        # Solve the game
        print(f"Solving with Value Iteration...")
        result = solver.solve(max_iterations=1000)
        total_time = time.time() - start_time
        
        # Print empty board
        print("\nEmpty board layout:")
        empty_board = tuple([0] * (k*k))
        game.print_board(empty_board)
        
        # Find optimal first move
        initial_state = game.get_initial_state()
        optimal_move = solver.play_optimal_move(initial_state)
        row, col = optimal_move // k, optimal_move % k
        
        print(f"\nRESULTS:")
        print(f"- Found {len(solver.values)} possible game states")
        print(f"- Converged after {result['iterations']} iterations")
        print(f"- Solution time: {total_time:.2f} seconds")
        print(f"- Optimal first move: Position {optimal_move} (row {row}, col {col})")
        
        # Create board with optimal move
        optimal_board = list(empty_board)
        optimal_board[optimal_move] = 1  # Player X (1)
        print("\nOptimal opening:")
        game.print_board(tuple(optimal_board))
        
        # Run simulations against random player
        num_simulations = 1000
        print(f"\nRunning {num_simulations} simulations against random player...")
        
        # Track results
        wins = draws = losses = 0
        total_steps = 0
        
        for i in range(num_simulations):
            # Use optimal policy for first player
            def optimal_policy(state):
                return solver.play_optimal_move(state)
            
            # Simulate game
            _, winner, steps = game.simulate_random_opponent(game.get_initial_state(), optimal_policy)
            total_steps += steps
            
            if winner == 1:  # player X (first player)
                wins += 1
            elif winner == 2:  # player O (second player)
                losses += 1
            else:  # Draw
                draws += 1
        
        # Print results
        print(f"\nSIMULATION RESULTS:")
        print(f"- Wins: {wins}/{num_simulations} ({wins/num_simulations*100:.1f}%)")
        print(f"- Draws: {draws}/{num_simulations} ({draws/num_simulations*100:.1f}%)")
        print(f"- Losses: {losses}/{num_simulations} ({losses/num_simulations*100:.1f}%)")
        print(f"- Average game length: {total_steps/num_simulations:.1f} moves")
        
        # Interpret results
        if wins > 0.95 * num_simulations:
            print("\nCONCLUSION: First player can FORCE A WIN with optimal play")
        elif wins + draws > 0.95 * num_simulations:
            print("\nCONCLUSION: First player can AVOID LOSING with optimal play")
        else:
            print("\nCONCLUSION: Even with optimal play, first player may lose to random play")

if __name__ == "__main__":
    # By default, analyze 3×3 and 4×4 boards
    # To analyze only specific sizes, pass them as arguments
    if len(sys.argv) > 1:
        k_values = [int(k) for k in sys.argv[1:]]
    else:
        k_values = [3, 4]
    
    analyze_tictactoe(k_values) 