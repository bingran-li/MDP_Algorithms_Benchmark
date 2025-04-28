#!/usr/bin/env python
"""
Benchmark Optimal Solution Comparison

This script demonstrates:
1. The calculation of optimal values using 10000 iterations
2. The improved convergence criterion for influence-tree-vi
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
from mdp_lib.algorithms import ValueIteration, calculate_optimal_values

# Set global random seed for reproducibility
SEED = 42


def run_optimal_solution_benchmark():
    """
    Compare different algorithms against an optimal reference solution
    """
    print("\nComparing algorithms against optimal solution...")
    print("=" * 80)
    
    # Initialize generator and create an MDP
    generator = MDPGenerator(seed=SEED)
    
    # Create a few different MDPs to test
    mdps = []
    
    # Small deterministic MDP
    mdp_small = generator.generate_mdp(n_states=20, m_actions=4, mdp_type="deterministic")
    mdp_small['metadata'] = {'name': 'Small Deterministic (20 states)'}
    mdps.append(mdp_small)
    
    # Medium stochastic MDP
    mdp_medium = generator.generate_mdp(n_states=50, m_actions=4, mdp_type="stochastic", sparsity=0.3)
    mdp_medium['metadata'] = {'name': 'Medium Stochastic (50 states)'}
    mdps.append(mdp_medium)
    
    # Large stochastic MDP
    mdp_large = generator.generate_mdp(n_states=100, m_actions=4, mdp_type="stochastic", sparsity=0.1)
    mdp_large['metadata'] = {'name': 'Large Stochastic (100 states)'}
    mdps.append(mdp_large)
    
    # Initialize benchmarker
    benchmarker = Benchmarker(seed=SEED)
    
    # Define algorithms to test
    algorithms = [
        ('value_iteration', {'rule': 'standard'}, 'Standard VI'),
        ('value_iteration', {'rule': 'gauss-seidel'}, 'Gauss-Seidel VI'),
        ('value_iteration', {'rule': 'influence-tree-vi', 'update_batch_size': 5}, 'Influence Tree VI (5)'),
        ('value_iteration', {'rule': 'influence-tree-vi', 'update_batch_size': 10}, 'Influence Tree VI (10)'),
        ('value_iteration', {'rule': 'random-vi', 'subset_size': 0.3}, 'Random VI (30%)'),
    ]
    
    # Results storage
    results = []
    
    # Test each algorithm on each MDP
    for mdp in mdps:
        mdp_name = mdp['metadata']['name']
        print(f"\nTesting on {mdp_name}")
        
        for alg_name, params, display_name in algorithms:
            print(f"\n  Testing {display_name}...")
            
            # Run the algorithm with optimal comparison
            result = benchmarker.run_benchmark_with_optimal_reference(mdp, alg_name, params)
            
            # Add display name and MDP info
            result['display_name'] = display_name
            result['mdp_name'] = mdp_name
            
            # Store result
            results.append(result)
    
    # Create summary table
    summary_rows = []
    for result in results:
        row = {
            'Algorithm': result['display_name'],
            'MDP': result['mdp_name'],
            'Iterations': result['iterations'],
            'Runtime (s)': result['runtime'],
            'Max Error': result['error_from_optimal'],
            'Mean Error': result['mean_error_from_optimal']
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Display summary table
    print("\n=== Summary Results ===")
    print(summary_df.to_string(index=False))
    
    # Create error vs runtime plot
    plt.figure(figsize=(10, 6))
    
    for mdp_name in summary_df['MDP'].unique():
        subset = summary_df[summary_df['MDP'] == mdp_name]
        plt.scatter(subset['Runtime (s)'], subset['Max Error'], 
                   label=mdp_name, alpha=0.7, s=100)
        
        # Add algorithm labels
        for _, row in subset.iterrows():
            plt.annotate(row['Algorithm'], 
                        (row['Runtime (s)'], row['Max Error']),
                        xytext=(7, 0), textcoords='offset points')
    
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Max Error from Optimal Solution')
    plt.title('Error vs. Runtime Trade-off')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    os.makedirs('code/results', exist_ok=True)
    fig_path = 'code/results/optimal_comparison.png'
    plt.savefig(fig_path)
    print(f"\nSaved error vs. runtime plot to '{fig_path}'")
    
    return results, summary_df


def analyze_influence_tree_vi_convergence():
    """
    Analyze the convergence behavior of the improved influence-tree-vi algorithm
    """
    print("\nAnalyzing Influence Tree VI convergence...")
    print("=" * 80)
    
    # Generate a medium-sized MDP
    generator = MDPGenerator(seed=SEED)
    mdp = generator.generate_mdp(n_states=100, m_actions=4, mdp_type="stochastic", sparsity=0.2)
    
    # Calculate optimal solution
    print("Calculating optimal solution...")
    optimal_values = calculate_optimal_values(mdp, iterations=10000, tolerance=1e-10)
    
    # Test with different configurations of influence-tree-vi
    configs = [
        {'update_batch_size': 5, 'name': 'Batch Size 5'},
        {'update_batch_size': 10, 'name': 'Batch Size 10'},
        {'update_batch_size': 20, 'name': 'Batch Size 20'},
    ]
    
    # Run each configuration and track convergence
    results = []
    
    for config in configs:
        print(f"\nTesting Influence Tree VI with {config['name']}...")
        
        # Create solver and run
        vi = ValueIteration(seed=SEED)
        params = {'rule': 'influence-tree-vi', 'update_batch_size': config['update_batch_size']}
        
        start_time = time.time()
        result = vi.solve(mdp, max_iterations=1000, tolerance=1e-6, **params)
        runtime = time.time() - start_time
        
        # Calculate error from optimal
        error = np.max(np.abs(result['values'] - optimal_values))
        
        print(f"  Completed in {runtime:.4f} seconds with {result['iterations']} iterations")
        print(f"  Error from optimal: {error:.6f}")
        
        # Store result with config info
        result['config_name'] = config['name']
        result['update_batch_size'] = config['update_batch_size']
        result['runtime'] = runtime
        result['error_from_optimal'] = error
        result['optimal_values'] = optimal_values
        
        results.append(result)
    
    # Plot convergence history
    plt.figure(figsize=(12, 6))
    
    for result in results:
        # Extract value history
        value_history = result['value_history']
        
        # Calculate error at each iteration
        errors = [np.max(np.abs(values - optimal_values)) for values in value_history]
        
        # Plot on log scale
        plt.semilogy(errors, label=f"{result['config_name']} ({result['iterations']} iterations)")
    
    plt.xlabel('Iterations')
    plt.ylabel('Max Error (log scale)')
    plt.title('Convergence of Influence Tree VI with Improved Criterion')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    fig_path = 'code/results/influence_tree_convergence.png'
    plt.savefig(fig_path)
    print(f"\nSaved convergence plot to '{fig_path}'")
    
    return results


if __name__ == "__main__":
    print("Optimal Solution Benchmark")
    print("=" * 80)
    
    # Part 1: Compare algorithms against optimal solution
    benchmark_results, summary = run_optimal_solution_benchmark()
    
    # Part 2: Analyze influence-tree-vi convergence
    convergence_results = analyze_influence_tree_vi_convergence()
    
    print("\nBenchmark completed!")
    print("=" * 80)