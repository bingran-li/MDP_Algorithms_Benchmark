#!/usr/bin/env python
"""
Enhanced Value Iteration Comparison and Tic-Tac-Toe Solver

This script demonstrates:
1. The standard Value Iteration vs RandomVI vs Influence Tree VI
2. The Tic-Tac-Toe game solver using Markov Game Processes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import from our mdp_lib package
from mdp_lib import MDPGenerator, Benchmarker
from mdp_lib.algorithms import ValueIteration
from mdp_lib.markov_games import TicTacToeGame, MGPValueIteration, play_tic_tac_toe_game, solve_tic_tac_toe_example

# Set global random seed for reproducibility
SEED = 42


def compare_value_iteration_approaches():
    """
    Compare the three Value Iteration approaches:
    1. Standard VI: Updates all states in each iteration
    2. RandomVI: Randomly selects a subset of states in each iteration
    3. Influence Tree VI: Uses an influence tree to select states to update
    """
    print("\nComparing Value Iteration Approaches...")
    print("=" * 80)
    
    # Initialize generator and MDPs
    generator = MDPGenerator(seed=SEED)
    
    # Create different types of MDPs for testing
    mdp_specs = [
        # Small MDPs
        {"n_states": 20, "m_actions": 4, "mdp_type": "stochastic", "sparsity": 0.3},
        {"n_states": 20, "m_actions": 4, "mdp_type": "deterministic"},
        
        # Medium MDPs
        {"n_states": 100, "m_actions": 4, "mdp_type": "stochastic", "sparsity": 0.1},
        {"n_states": 100, "m_actions": 4, "mdp_type": "deterministic"},
        
        # Large MDPs
        {"n_states": 500, "m_actions": 4, "mdp_type": "stochastic", "sparsity": 0.05},
        {"n_states": 500, "m_actions": 4, "mdp_type": "deterministic"}
    ]
    
    mdps = []
    print("\nGenerating MDPs...")
    for i, spec in enumerate(mdp_specs):
        print(f"  MDP {i+1}: {spec['n_states']} states, "
              f"{spec['mdp_type']}, "
              f"sparsity={spec.get('sparsity', 'N/A')}")
        mdp = generator.generate_mdp(**spec)
        mdp['metadata'] = {
            'id': i,
            'spec': spec,
            'seed': SEED + i
        }
        mdps.append(mdp)
    
    # Value Iteration configurations to test
    vi_configs = [
        # Standard VI
        {"rule": "standard", "name": "Standard VI"},
        
        # RandomVI with different subset sizes
        {"rule": "random-vi", "subset_size": 0.1, "name": "RandomVI (10%)"},
        {"rule": "random-vi", "subset_size": 0.3, "name": "RandomVI (30%)"},
        {"rule": "random-vi", "subset_size": 0.5, "name": "RandomVI (50%)"},
        
        # Influence Tree VI with different batch sizes
        {"rule": "influence-tree-vi", "update_batch_size": 5, "name": "Influence Tree VI (5)"},
        {"rule": "influence-tree-vi", "update_batch_size": 10, "name": "Influence Tree VI (10)"},
        {"rule": "influence-tree-vi", "update_batch_size": 20, "name": "Influence Tree VI (20)"}
    ]
    
    # Results storage
    results = []
    
    # Run all configurations on all MDPs
    print("\nRunning Value Iteration approaches...")
    for mdp_idx, mdp in enumerate(mdps):
        mdp_info = mdp['metadata']['spec']
        n_states = mdp_info['n_states']
        mdp_type = mdp_info['mdp_type']
        sparsity = mdp_info.get('sparsity', 'N/A')
        
        print(f"\nMDP {mdp_idx+1}: {n_states} states, {mdp_type}, sparsity={sparsity}")
        
        for config in vi_configs:
            vi = ValueIteration(seed=SEED+mdp_idx)
            
            # Extract parameters
            params = {k: v for k, v in config.items() if k != "name"}
            
            print(f"  Running {config['name']}...")
            start_time = time.time()
            result = vi.solve(mdp, max_iterations=1000, tolerance=1e-6, **params)
            runtime = time.time() - start_time
            
            # Store results
            result_data = {
                'mdp_idx': mdp_idx,
                'n_states': n_states,
                'mdp_type': mdp_type,
                'sparsity': sparsity,
                'algorithm': config['name'],
                'rule': params['rule'],
                'iterations': result['iterations'],
                'runtime': runtime,
                'value_norm': np.linalg.norm(result['values']),
                'mean_value': np.mean(result['values']),
                'subset_size': params.get('subset_size', None),
                'update_batch_size': params.get('update_batch_size', None)
            }
            results.append(result_data)
            
            print(f"    Completed in {runtime:.4f} seconds, {result['iterations']} iterations")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Display summary tables
    print("\n=== Summary Results ===")
    
    # Efficiency table
    print("\nEfficiency Comparison (Runtime):")
    runtime_table = df.pivot_table(
        values='runtime',
        index=['n_states', 'mdp_type'],
        columns='algorithm',
        aggfunc='mean'
    )
    print(runtime_table)
    
    # Iterations table
    print("\nIterations Comparison:")
    iterations_table = df.pivot_table(
        values='iterations',
        index=['n_states', 'mdp_type'],
        columns='algorithm',
        aggfunc='mean'
    )
    print(iterations_table)
    
    # Calculate speedup vs standard VI
    print("\nSpeedup vs Standard VI:")
    speedup_table = runtime_table.copy()
    std_vi_col = speedup_table["Standard VI"]
    for col in speedup_table.columns:
        if col != "Standard VI":
            speedup_table[col] = std_vi_col / speedup_table[col]
    speedup_table["Standard VI"] = 1.0
    print(speedup_table)
    
    # Plot runtime comparison
    plt.figure(figsize=(12, 8))
    for algorithm in df['algorithm'].unique():
        subset = df[df['algorithm'] == algorithm]
        
        # Group by MDP size
        sizes = sorted(subset['n_states'].unique())
        runtimes = []
        
        for size in sizes:
            mean_runtime = subset[subset['n_states'] == size]['runtime'].mean()
            runtimes.append(mean_runtime)
        
        plt.plot(sizes, runtimes, 'o-', label=algorithm)
    
    plt.xlabel('MDP Size (number of states)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison of Value Iteration Approaches')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('code/results', exist_ok=True)
    plt.savefig('code/results/vi_comparison.png')
    print("\nSaved runtime comparison plot to 'code/results/vi_comparison.png'")
    
    # Return results DataFrame for further analysis
    return df


def solve_tictactoe():
    """
    Demonstrate the Tic-Tac-Toe game solver using Markov Game Processes
    """
    print("\nDemonstrating Tic-Tac-Toe Game Solver...")
    print("=" * 80)
    
    # Solve Tic-Tac-Toe
    solver = solve_tic_tac_toe_example()
    
    # Ask if the user wants to play
    play_game = input("\nWould you like to play a game of Tic-Tac-Toe against the AI? (y/n): ")
    if play_game.lower() in ['y', 'yes']:
        # Ask which player they want to be
        player_input = input("Do you want to play as X (first) or O (second)? (X/O): ")
        player_human = 0 if player_input.upper() == 'X' else 1
        
        # Play the game
        play_tic_tac_toe_game(solver, player_human)
    
    return solver


if __name__ == "__main__":
    print("Enhanced Value Iteration and Markov Game Processes Demo")
    print("=" * 80)
    
    # Part 1: Compare Value Iteration approaches
    results_df = compare_value_iteration_approaches()
    
    # Part 2: Demonstrate Tic-Tac-Toe solver
    solver = solve_tictactoe()
    
    print("\nDemo completed!")
    print("=" * 80) 