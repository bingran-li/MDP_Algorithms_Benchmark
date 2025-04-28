#!/usr/bin/env python
"""
Concise analysis of k×k Tic-Tac-Toe against a random player
"""
import numpy as np
import sys
import os
from tabulate import tabulate

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import our Tic-Tac-Toe implementation
from mdp_lib.markov_games import TicTacToeGame, MGPValueIteration

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

def analyze_brief(k_values=[3, 4]):
    """Concise analysis of k×k Tic-Tac-Toe"""
    results = []
    
    print(f"Analyzing {len(k_values)} board sizes: {', '.join(str(k) for k in k_values)}")
    
    for k in k_values:
        print(f"\nSolving {k}×{k} Tic-Tac-Toe...")
        
        # Create game and solver
        game = TicTacToeGame(board_size=k, seed=SEED)
        solver = MGPValueIteration(game, seed=SEED)
        
        # Solve the game
        solver.solve(max_iterations=1000)
        
        # Find optimal first move
        initial_state = game.get_initial_state()
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
        
        # Run simulations against random player
        num_simulations = 1000
        print(f"Running {num_simulations} simulations against random player...")
        
        # Define policy function
        def optimal_policy(state):
            return solver.play_optimal_move(state)
        
        # Track results
        wins = draws = losses = 0
        
        for _ in range(num_simulations):
            _, winner, _ = game.simulate_random_opponent(game.get_initial_state(), optimal_policy)
            if winner == 1:  # player X (first player)
                wins += 1
            elif winner == 2:  # player O (second player)
                losses += 1
            else:  # Draw
                draws += 1
                
        win_rate = wins / num_simulations * 100
        
        results.append({
            "board_size": k,
            "optimal_move": optimal_move,
            "position": f"({row}, {col})",
            "type": position_type,
            "win_rate": win_rate,
            "draw_rate": draws / num_simulations * 100,
            "loss_rate": losses / num_simulations * 100
        })
        
    # Create summary table
    table_data = []
    for r in results:
        can_always_win = "Yes" if r["win_rate"] > 99 else "No"
        table_data.append([
            f"{r['board_size']}×{r['board_size']}",
            f"{r['optimal_move']} {r['position']}",
            r['type'].capitalize(),
            f"{r['win_rate']:.1f}%",
            f"{r['draw_rate']:.1f}%",
            f"{r['loss_rate']:.1f}%",
            can_always_win
        ])
    
    headers = ["Board Size", "Optimal First Move", "Position Type", 
              "Win Rate", "Draw Rate", "Loss Rate", "Always Win?"]
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Key findings
    print("\nKey Findings:")
    for r in results:
        k = r["board_size"]
        print(f"\n{k}×{k} Board:")
        print(f"- Optimal first move: Position {r['optimal_move']} {r['position']} ({r['type']})")
        
        if r["win_rate"] > 99:
            print(f"- First player CAN ALWAYS WIN against a random player")
        elif r["win_rate"] + r["draw_rate"] > 99:
            print(f"- First player CANNOT LOSE with optimal play against a random player")
        else:
            print(f"- First player can lose even with optimal play ({r['loss_rate']:.1f}% loss rate)")
    
    print("\nConclusion:")
    for r in results:
        k = r["board_size"]
        if r["win_rate"] > 99:
            print(f"- {k}×{k}: First player has a guaranteed winning strategy")
        else:
            print(f"- {k}×{k}: Optimal play does not guarantee a win against a random opponent")

if __name__ == "__main__":
    analyze_brief() 