#!/usr/bin/env python
"""
Comprehensive MDP Algorithm Comparison Script

This script demonstrates the improved MDP benchmarking capabilities,
including standardized parameter naming, multiple runs for statistical analysis,
improved DataFrame comparison tables, and visualization tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import itertools
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import from our mdp_lib package
from mdp_lib import MDPGenerator, AlgorithmFactory, Benchmarker

# Set global random seed for reproducibility
SEED = 123456


def run_comprehensive_benchmark():
    """
    Run a comprehensive benchmark of all implemented algorithms with various configurations
    """
    print("\nRunning comprehensive MDP algorithm benchmark...")
    print("=" * 80)
    print(f"Using random seed: {SEED}")
    
    # Initialize benchmarker
    benchmarker = Benchmarker(seed=SEED)
    
    # Define MDP specifications
    mdp_specs = [
        (5, 'stochastic'),    # Very small stochastic
        (10, 'stochastic'),   # Small stochastic
        (20, 'stochastic'),   # Medium-small stochastic
        (40, 'stochastic'),   # Medium stochastic
        (60, 'stochastic'),   # Medium-large stochastic
        (80, 'stochastic'),   # Large stochastic
        (100, 'stochastic'),  # Very large stochastic
        (200, 'stochastic'),  # Extra large stochastic
        (5, 'deterministic'),   # Very small deterministic
        (10, 'deterministic'),  # Small deterministic
        (20, 'deterministic'),  # Medium-small deterministic
        (40, 'deterministic'),  # Medium deterministic
        (60, 'deterministic'),  # Medium-large deterministic
        (80, 'deterministic'),  # Large deterministic
        (100, 'deterministic'),  # Very large deterministic
        (200, 'deterministic')  # Extra large deterministic
    ]
    
    # Prepare MDPs
    print("\nPreparing MDPs...")
    mdps = benchmarker.prepare_mdps(mdp_specs)
    print(f"Created {len(mdps)} MDPs for benchmarking")
    
    # Calculate optimal values for each MDP using high-precision solver
    print("\nCalculating optimal values for each MDP using high-precision solver...")
    optimal_values = {}
    
    for i, mdp in enumerate(mdps):
        print(f"  Processing MDP {i+1}/{len(mdps)} ({mdp['n_states']} states, {mdp['type']})")
        mdp_id = (mdp['n_states'], mdp['type'])
        optimal_values[mdp_id] = benchmarker.compute_optimal_solution(mdp)
    
    print(f"Optimal values calculated for {len(optimal_values)} MDPs")
    
    # Define algorithms and parameter grid
    algorithms = ['policy_iteration', 'value_iteration', 'simplex']
    param_grid = {
        'policy_iteration': {'rule': ['standard', 'modified']},
        'value_iteration': {'rule': ['standard', 'gauss-seidel', 'random-vi', 'influence-tree-vi', 'rp-cyclic-vi']},
        'simplex': {'rule': ['bland', 'largest_coefficient', 'steepest_edge']}
    }
    
    # Run benchmark with multiple runs per configuration for statistical analysis
    n_runs = 3
    print(f"\nRunning benchmark with {n_runs} runs per configuration...")
    print(f"Algorithms: {algorithms}")
    print(f"Parameters: {param_grid}")
    
    start_time = time.time()
    results = benchmarker.benchmark_algorithms(algorithms, param_grid, n_runs=n_runs)
    total_time = time.time() - start_time
    
    print(f"\nBenchmark completed in {total_time:.2f} seconds")
    print(f"Total configurations tested: {len(results)}")
    
    # Add error from optimal solution to each result
    print("\nCalculating error from optimal solution for each result...")
    for result in results:
        mdp_id = (result['n_states'], result['mdp_type'])
        if mdp_id in optimal_values:
            # Calculate error metrics
            opt_values = optimal_values[mdp_id]
            result_values = result['values']
            
            # Different error metrics
            result['error_from_optimal'] = np.max(np.abs(result_values - opt_values))
            result['mean_error_from_optimal'] = np.mean(np.abs(result_values - opt_values))
            result['rmse_from_optimal'] = np.sqrt(np.mean(np.square(result_values - opt_values)))
            
            # Save optimal values reference
            result['optimal_values'] = opt_values
    
    # Convert results to DataFrame for analysis
    print("\nConverting results to DataFrame...")
    df = benchmarker.results_to_dataframe()
    print(f"DataFrame shape: {df.shape}")
    
    # Save results to files
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.json')
    csv_file = os.path.join(output_dir, f'results_table_{timestamp}.csv')
    
    print(f"\nSaving results to {results_file}")
    benchmarker.save_results(results_file)
    
    print(f"Saving DataFrame to {csv_file}")
    df.to_csv(csv_file, index=False)
    
    return benchmarker


def create_comparison_tables(benchmarker):
    """
    Create various comparison tables to analyze the benchmark results
    
    Args:
        benchmarker: The benchmarker object with results
        
    Returns:
        dict: Dictionary of comparison tables
    """
    print("\nCreating comparison tables...")
    
    tables = {}
    
    # Get the DataFrame and check column names to avoid KeyError
    df = benchmarker.df_results
    
    # Determine the runtime column name - it could be 'runtime_seconds' or 'runtime'
    runtime_col = 'runtime_seconds' if 'runtime_seconds' in df.columns else 'runtime'
    
    # 1. Standard comparison table, grouped by algorithm and rule
    print("  - Creating standard comparison table...")
    standard = benchmarker.create_comparison_table(
        group_by=['algorithm', 'param_rule', 'n_states', 'mdp_type'],
        metrics=['iterations', runtime_col, 'mean_value'],
        sort_by=['algorithm', 'param_rule', 'n_states']
    )
    tables['standard'] = standard
    
    # 2. Pivot table comparing algorithms across state sizes
    print("  - Creating pivot table by state size...")
    pivot_states = benchmarker.create_pivot_table(
        values=['iterations', runtime_col],
        index=['n_states', 'mdp_type'],
        columns=['algorithm', 'param_rule'],
        aggfunc='mean'
    )
    tables['pivot_states'] = pivot_states
    
    # 3. Comparison focusing on algorithmic efficiency
    print("  - Creating efficiency comparison table...")
    efficiency = benchmarker.create_comparison_table(
        group_by=['algorithm', 'param_rule'],
        metrics=['iterations', runtime_col],
        sort_by=f'{runtime_col}_mean',
        ascending=True
    )
    tables['efficiency'] = efficiency
    
    # 4. Comparison by MDP type
    print("  - Creating MDP type comparison table...")
    mdp_type = benchmarker.create_comparison_table(
        group_by=['mdp_type', 'algorithm', 'param_rule'],
        metrics=['iterations', runtime_col],
        sort_by=['mdp_type', f'{runtime_col}_mean'],
        ascending=[True, True]
    )
    tables['mdp_type'] = mdp_type
    
    return tables


def print_tables(tables):
    """Print the comparison tables"""
    
    # Loop through each table and print
    for name, table in tables.items():
        print(f"\n{name.upper()} Comparison Table:")
        print("-" * 100)
        
        if isinstance(table, pd.DataFrame):
            # For wide tables, limit the columns to display
            pd.set_option('display.max_columns', 12)
            pd.set_option('display.width', 1000)
            print(table.to_string())
        else:
            print(table)
            
        print("-" * 100)
    
    print("\n")


def create_visualizations(benchmarker):
    """Create visualizations of the benchmark results"""
    print("\nCreating visualizations...")
    
    # Get the DataFrame and check column names to avoid KeyError
    df = benchmarker.df_results
    
    # Determine the runtime column name - it could be 'runtime_seconds' or 'runtime'
    runtime_col = 'runtime_seconds' if 'runtime_seconds' in df.columns else 'runtime'
    
    # 1. Performance comparison by state size for different algorithms
    print("  - Creating performance comparison by state size...")
    benchmarker.plot_performance_comparison(
        metric=runtime_col,
        by='n_states',
        figsize=(12, 8)
    )
    
    # 2. Iteration count comparison by state size
    print("  - Creating iteration count comparison...")
    benchmarker.plot_performance_comparison(
        metric='iterations',
        by='n_states',
        figsize=(12, 8)
    )
    
    # 3. Box plot of runtime by algorithm
    print("  - Creating runtime boxplot...")
    benchmarker.plot_boxplot_comparison(
        metric=runtime_col,
        group_by=['algorithm', 'param_rule'],
        figsize=(14, 8)
    )
    
    # 4. Convergence plots for selected algorithms on small MDPs
    print("  - Creating convergence plots...")
    
    # Define a filter function for small MDPs
    def small_mdp_filter(result, _):
        return (result['n_states'] == 10 and 
                result['mdp_type'] == 'stochastic' and
                result['run_idx'] == 0)  # Only first run
    
    # Use the original convergence plot function
    print("    Standard convergence plot (using difference from 'optimal' solution):")
    benchmarker.plot_convergence(
        subset=small_mdp_filter,
        figsize=(12, 8)
    )
    
    # Use the improved convergence plot with Bellman error
    print("    Improved convergence plot (using Bellman error):")
    benchmarker.plot_improved_convergence(
        subset=small_mdp_filter,
        figsize=(12, 8),
        use_bellman_error=True
    )
    
    # Also create convergence plots for larger MDPs to see behavior more clearly
    def medium_mdp_filter(result, _):
        return (result['n_states'] == 40 and 
                result['mdp_type'] == 'stochastic' and
                result['run_idx'] == 0)  # Only first run
    
    print("    Improved convergence plot for medium-sized MDPs:")
    benchmarker.plot_improved_convergence(
        subset=medium_mdp_filter,
        figsize=(12, 8),
        use_bellman_error=True
    )

    # Also create convergence plots for larger MDPs to see behavior more clearly
    def medium_mdp_filter(result, _):
        return (result['n_states'] == 40 and 
                result['mdp_type'] == 'deterministic' and
                result['run_idx'] == 0)  # Only first run
    
    print("    Improved convergence plot for medium-sized MDPs:")
    benchmarker.plot_improved_convergence(
        subset=medium_mdp_filter,
        figsize=(12, 8),
        use_bellman_error=True
    )
    
    # 5. NEW: Iteration count comparison between dense/sparse MDPs for each method
    print("  - Creating stochastic/deterministic iteration count comparison...")
    plt.figure(figsize=(14, 8))
    
    # Define MDP types for comparison
    mdp_types = ['deterministic', 'stochastic'] 
    
    # Get algorithm-parameter combinations from the results DataFrame
    unique_algs = df['algorithm'].unique()
    alg_params = []
    
    for alg in unique_algs:
        alg_rules = df[df['algorithm'] == alg]['param_rule'].unique()
        for rule in alg_rules:
            alg_params.append((alg, rule))
    
    # Setup plot
    bar_width = 0.35
    index = np.arange(len(alg_params))
    
    # Prepare data for plotting
    means_by_type = {mdp_type: [] for mdp_type in mdp_types}
    labels = []
    
    # Filter for state size 200 to match what we do in convergence rate plot
    state_size = 200
    
    # Get iteration means for each algorithm and MDP type
    for alg, rule in alg_params:
        labels.append(f"{alg}\n({rule})")
        
        for mdp_type in mdp_types:
            # Create a mask to filter by appropriate conditions
            if mdp_type == 'deterministic':
                mask = ((df['algorithm'] == alg) & 
                        (df['param_rule'] == rule) &
                        (df['n_states'] == state_size) &
                        (df['mdp_type'] == 'deterministic')) 
            else: 
                mask = ((df['algorithm'] == alg) & 
                        (df['param_rule'] == rule) &
                        (df['n_states'] == state_size) &
                        (df['mdp_type'] == 'stochastic')) 
            
            subset = df[mask]
            
            if not subset.empty:
                means_by_type[mdp_type].append(subset['iterations'].mean())
            else:
                means_by_type[mdp_type].append(0)
    
    # Plot bars
    for i, mdp_type in enumerate(mdp_types):
        plt.bar(index + i*bar_width, means_by_type[mdp_type], bar_width, 
                label=f"{mdp_type.capitalize()} MDP",
                alpha=0.7)
    
    plt.xlabel('Algorithm (rule)')
    plt.ylabel('Average Number of Iterations')
    plt.title('Iteration Count Comparison: stochastic vs deterministic MDPs (n_states=200)')
    plt.xticks(index + bar_width/2, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # 6. NEW: Runtime by state size for dense and sparse MDPs
    print("  - Creating runtime by state size for dense/sparse MDPs...")
    state_sizes = [5, 20, 40, 60, 80, 100, 200]
    
    for mdp_type in mdp_types:
        plt.figure(figsize=(12, 8))
        
        # Get data for each algorithm/rule combination
        for alg, rule in alg_params:
            # Filter for this algorithm and rule
            mask = (df['algorithm'] == alg) & (df['param_rule'] == rule)
            alg_df = df[mask]
            
            # Filter for current MDP type
            if mdp_type == 'sparse':
                type_mask = (alg_df['mdp_type'] == 'deterministic')
            else:  # dense
                type_mask = (alg_df['mdp_type'] == 'stochastic')
            
            type_df = alg_df[type_mask]
            
            # Get runtime for each state size
            x_values = []
            y_values = []
            
            for size in state_sizes:
                size_subset = type_df[type_df['n_states'] == size]
                if not size_subset.empty:
                    x_values.append(size)
                    y_values.append(size_subset[runtime_col].mean())
            
            if x_values:  # Only plot if we have data
                plt.plot(x_values, y_values, 'o-', 
                         label=f"{alg} ({rule})",
                         linewidth=2, markersize=8)
        
        plt.xlabel('Number of States')
        plt.ylabel('Runtime (seconds)')
        plt.title(f'Runtime Comparison by State Size ({mdp_type.capitalize()} MDPs)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 7. NEW: Plot convergence rate e_k+1/e_k
    print("  - Creating convergence rate plot...")
    
    # Filter for algorithms where we want to analyze convergence rate
    convergence_algs = list(unique_algs)  # Use all available algorithms
    
    plt.figure(figsize=(12, 8))
    
    # Use only MDPs with n_states=200 for better visualization
    def convergence_rate_filter(result, _):
        return (result['n_states'] == 200 and 
                result['algorithm'] in convergence_algs and
                result['run_idx'] == 0)  # Only first run
    
    # Filter and prepare data
    conv_results = [r for i, r in enumerate(benchmarker.results) 
                  if convergence_rate_filter(r, i)]
    
    for result in conv_results:
        alg = result['algorithm']
        params = result['parameters']
        params_str = '_'.join(f"{k}:{v}" for k, v in params.items())
        label = f"{alg} ({params_str}) - {result['mdp_type']}"
        
        # Extract convergence history
        conv_history = result.get('conv_history', [])
        
        # Calculate convergence rate e_k+1/e_k
        if len(conv_history) > 2:  # Need at least 3 points
            rates = [conv_history[i+1]/conv_history[i] if conv_history[i] > 1e-10 else 0 
                    for i in range(len(conv_history)-1)]
            
            # Plot convergence rate
            plt.plot(range(1, len(rates)+1), rates, 'o-', label=label, alpha=0.7)
    
    plt.axhline(y=benchmarker.results[0].get('parameters', {}).get('gamma', 0.9), 
                color='r', linestyle='--', label='Discount Factor (γ)')
    plt.xlabel('Iterations')
    plt.ylabel('Convergence Rate (e_k+1/e_k)')
    plt.title('Convergence Rate Analysis (n_states=200)')
    plt.ylim(0, 1.1)  # Rates are typically between 0 and 1
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_analysis_summary(benchmarker):
    """Print a summary analysis of the benchmark results"""
    print("\nBenchmark Analysis Summary")
    print("=" * 80)
    
    # Convert to DataFrame if needed
    if benchmarker.df_results is None:
        benchmarker.results_to_dataframe()
    
    df = benchmarker.df_results
    
    # Determine the runtime column name - it could be 'runtime_seconds' or 'runtime'
    runtime_col = 'runtime_seconds' if 'runtime_seconds' in df.columns else 'runtime'
    
    # 1. Best algorithm by runtime
    print("1. Best algorithm by average runtime:")
    runtime_by_alg = df.groupby(['algorithm', 'param_rule'])[runtime_col].mean().sort_values()
    for i, ((alg, rule), runtime) in enumerate(runtime_by_alg.items()):
        print(f"   {i+1}. {alg} ({rule}): {runtime:.6f} seconds")
    
    print("\n2. Best algorithm by iteration count:")
    iter_by_alg = df.groupby(['algorithm', 'param_rule'])['iterations'].mean().sort_values()
    for i, ((alg, rule), iters) in enumerate(iter_by_alg.items()):
        print(f"   {i+1}. {alg} ({rule}): {iters:.2f} iterations")
    
    print("\n3. Performance across different MDP types:")
    for mdp_type in df['mdp_type'].unique():
        subset = df[df['mdp_type'] == mdp_type]
        runtime_by_type = subset.groupby(['algorithm', 'param_rule'])[runtime_col].mean().sort_values()
        print(f"   {mdp_type.capitalize()} MDPs - Best algorithm: "
              f"{runtime_by_type.index[0][0]} ({runtime_by_type.index[0][1]})"
              f" - {runtime_by_type.iloc[0]:.6f} seconds")
    
    print("\n4. Performance across different state space sizes:")
    for size in sorted(df['n_states'].unique()):
        subset = df[df['n_states'] == size]
        runtime_by_size = subset.groupby(['algorithm', 'param_rule'])[runtime_col].mean().sort_values()
        print(f"   {size} states - Best algorithm: "
              f"{runtime_by_size.index[0][0]} ({runtime_by_size.index[0][1]})"
              f" - {runtime_by_size.iloc[0]:.6f} seconds")
    
    print("\n5. Simplex algorithm pivot rule comparison:")
    simplex_df = df[df['algorithm'] == 'simplex']
    if not simplex_df.empty:
        pivot_compare = simplex_df.groupby('param_rule')[runtime_col].mean().sort_values()
        for rule, runtime in pivot_compare.items():
            print(f"   {rule}: {runtime:.6f} seconds")
    else:
        print("   No simplex algorithm results found")
    
    print("\n6. Overall recommendation:")
    best_alg, best_rule = runtime_by_alg.index[0]
    print(f"   The best overall algorithm is {best_alg} with {best_rule} rule")
    print(f"   Average runtime: {runtime_by_alg.iloc[0]:.6f} seconds")
    
    # Get average speedup between best and worst algorithm
    worst_alg, worst_rule = runtime_by_alg.index[-1]
    speedup = runtime_by_alg.iloc[-1] / runtime_by_alg.iloc[0]
    print(f"   Speedup over worst algorithm ({worst_alg} with {worst_rule}): {speedup:.2f}x")
    
    print("=" * 80)


def run_sparsity_benchmark():
    """
    Run a benchmark specifically to test algorithm performance on MDPs with varying sparsity levels
    """
    print("\nRunning MDP sparsity benchmark...")
    print("=" * 80)
    print(f"Using random seed: {SEED}")
    
    # Initialize benchmarker
    benchmarker = Benchmarker(seed=SEED)
    
    # Define MDP specifications with varying sparsity levels
    # We'll create stochastic MDPs with different levels of sparsity
    sparsity_levels = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    state_sizes = [50, 100, 200]
    
    mdp_specs = []
    for size in state_sizes:
        for sparsity in sparsity_levels:
            # Add sparsity level to the MDP specification
            mdp_specs.append((size, 'stochastic', sparsity))
    
    # Prepare MDPs
    print("\nPreparing MDPs with different sparsity levels...")
    
    # We need to prepare MDPs manually since we need to set sparsity
    mdps = []
    mdp_gen = MDPGenerator(seed=SEED)
    
    for spec in mdp_specs:
        if len(spec) == 3:
            n_states, mdp_type, sparsity = spec
            mdp = mdp_gen.generate_mdp(
                n_states=n_states,
                m_actions=4,  # Using 4 actions for all MDPs
                mdp_type=mdp_type,
                sparsity=sparsity
            )
            # Add metadata to track sparsity level
            mdp['sparsity'] = sparsity
            mdps.append(mdp)
    
    print(f"Created {len(mdps)} MDPs with different sparsity levels for benchmarking")
    
    # Define subset of algorithms to test (choose only the most relevant ones)
    algorithms = ['value_iteration']
    param_grid = {
        'value_iteration': {'rule': ['standard', 'gauss-seidel', 'random-vi', 
                                     'influence-tree-vi', 'rp-cyclic-vi']}
    }
    
    # Run benchmark with one run per configuration for efficiency
    n_runs = 1
    print(f"\nRunning sparsity benchmark with {n_runs} runs per configuration...")
    print(f"Algorithms: {algorithms}")
    print(f"Parameters: {param_grid}")
    
    start_time = time.time()
    
    # We need custom benchmark function because we want to track sparsity
    results = []
    
    # Get all algorithm configurations
    alg_configs = []
    for alg in algorithms:
        param_combinations = list(
            dict(zip(param_grid[alg].keys(), values))
            for values in itertools.product(*param_grid[alg].values())
        )
        for params in param_combinations:
            alg_configs.append((alg, params))
    
    # Run benchmarks
    for mdp_idx, mdp in enumerate(mdps):
        for alg_name, params in alg_configs:
            algorithm = AlgorithmFactory.create(alg_name, seed=SEED)
            
            for run_idx in range(n_runs):
                print(f"  Running {alg_name} with {params} on MDP {mdp_idx+1}/{len(mdps)} (sparsity={mdp['sparsity']})")
                
                # Solve MDP and get results
                result = algorithm.solve(mdp, **params)
                
                # Add metadata
                result['algorithm'] = alg_name
                result['parameters'] = params
                result['n_states'] = mdp['n_states']
                result['mdp_type'] = mdp['type']
                result['sparsity'] = mdp['sparsity']
                result['mdp_idx'] = mdp_idx
                result['run_idx'] = run_idx
                
                # Calculate average value
                result['mean_value'] = np.mean(result['values'])
                
                results.append(result)
    
    total_time = time.time() - start_time
    
    print(f"\nSparsity benchmark completed in {total_time:.2f} seconds")
    print(f"Total configurations tested: {len(results)}")
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame([
        {
            'algorithm': r['algorithm'],
            **{'param_' + k: v for k, v in r['parameters'].items()},
            'n_states': r['n_states'],
            'mdp_type': r['mdp_type'],
            'sparsity': r['sparsity'],
            'mdp_idx': r['mdp_idx'],
            'run_idx': r['run_idx'],
            'iterations': r['iterations'],
            'runtime': r['runtime'],
            'mean_value': r.get('mean_value', 0)
        }
        for r in results
    ])
    
    # Save results to files
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f'sparsity_results_{timestamp}.csv')
    
    print(f"Saving sparsity results to {csv_file}")
    df.to_csv(csv_file, index=False)
    
    return df, results


def create_sparsity_visualizations(df, results):
    """Create visualizations specifically for the sparsity benchmark results"""
    print("\nCreating sparsity visualizations...")
    
    # 1. Iteration count vs sparsity level
    print("  - Creating iteration count vs sparsity plot...")
    plt.figure(figsize=(14, 8))
    
    # Group by algorithm rule and sparsity
    for rule in df['param_rule'].unique():
        # Filter for just this rule
        rule_df = df[df['param_rule'] == rule]
        
        # Set up a line for each state size
        for n_states in sorted(df['n_states'].unique()):
            subset = rule_df[rule_df['n_states'] == n_states]
            
            # Group by sparsity and get mean iterations
            sparsity_values = []
            iterations = []
            
            for sparsity, group in subset.groupby('sparsity'):
                sparsity_values.append(sparsity)
                iterations.append(group['iterations'].mean())
            
            # Sort by sparsity for proper line plot
            sort_idx = np.argsort(sparsity_values)
            sparsity_values = [sparsity_values[i] for i in sort_idx]
            iterations = [iterations[i] for i in sort_idx]
            
            plt.plot(sparsity_values, iterations, 'o-', 
                     label=f"{rule} (n={n_states})")
    
    plt.xlabel('Sparsity Level')
    plt.ylabel('Number of Iterations')
    plt.title('Effect of Sparsity on Iteration Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Runtime vs sparsity level
    print("  - Creating runtime vs sparsity plot...")
    plt.figure(figsize=(14, 8))
    
    # Group by algorithm rule and sparsity
    for rule in df['param_rule'].unique():
        # Filter for just this rule
        rule_df = df[df['param_rule'] == rule]
        
        # Use only n_states=200 for better visibility
        subset = rule_df[rule_df['n_states'] == 200]
        
        # Group by sparsity and get mean runtime
        sparsity_values = []
        runtimes = []
        
        for sparsity, group in subset.groupby('sparsity'):
            sparsity_values.append(sparsity)
            runtimes.append(group['runtime'].mean())
        
        # Sort by sparsity for proper line plot
        sort_idx = np.argsort(sparsity_values)
        sparsity_values = [sparsity_values[i] for i in sort_idx]
        runtimes = [runtimes[i] for i in sort_idx]
        
        plt.plot(sparsity_values, runtimes, 'o-', linewidth=2, 
                 label=f"{rule}")
    
    plt.xlabel('Sparsity Level')
    plt.ylabel('Runtime (seconds)')
    plt.title('Effect of Sparsity on Runtime (n_states=200)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Bar chart comparing algorithms at different sparsity levels
    print("  - Creating algorithm comparison barplot for different sparsity levels...")
    
    # We'll create one plot for each state size
    for n_states in sorted(df['n_states'].unique()):
        plt.figure(figsize=(14, 8))
        
        # Filter for this state size
        state_df = df[df['n_states'] == n_states]
        
        # Get all rules and sparsity levels
        rules = sorted(state_df['param_rule'].unique())
        sparsity_levels = sorted(state_df['sparsity'].unique())
        
        # Set up the plot
        x = np.arange(len(sparsity_levels))
        width = 0.8 / len(rules)  # Width of bars so they fit side by side
        
        # Create bars for each rule
        for i, rule in enumerate(rules):
            # Filter for just this rule
            rule_data = []
            for sparsity in sparsity_levels:
                subset = state_df[(state_df['param_rule'] == rule) & 
                                  (state_df['sparsity'] == sparsity)]
                if not subset.empty:
                    rule_data.append(subset['iterations'].mean())
                else:
                    rule_data.append(0)
            
            # Position the bars
            pos = x - 0.4 + (i + 0.5) * width
            
            plt.bar(pos, rule_data, width, label=rule)
        
        # Add labels and legend
        plt.xlabel('Sparsity Level')
        plt.ylabel('Number of Iterations')
        plt.title(f'Iteration Count by Algorithm at Different Sparsity Levels (n_states={n_states})')
        plt.xticks(x, [f"{s:.2f}" for s in sparsity_levels], rotation=45)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    # 4. Convergence rate vs sparsity
    print("  - Creating convergence rate vs sparsity plot...")
    
    # Create a plot for n_states = 200
    plt.figure(figsize=(14, 8))
    
    # For each rule and sparsity level, calculate average convergence rate
    for rule in df['param_rule'].unique():
        # Get all results for this rule with n_states = 200
        rule_results = [r for r in results if r['parameters'].get('rule') == rule 
                      and r['n_states'] == 200]
        
        sparsity_values = []
        conv_rates = []
        
        for result in rule_results:
            # Get convergence history
            conv_history = result.get('conv_history', [])
            
            # Calculate convergence rate if we have enough data points
            if len(conv_history) > 2:
                # Calculate average ratio of e_{k+1}/e_k over last 80% of iterations
                start_idx = int(len(conv_history) * 0.2)  # Skip first 20%
                ratios = [conv_history[i+1]/conv_history[i] 
                         for i in range(start_idx, len(conv_history)-1) 
                         if conv_history[i] > 1e-10]
                
                if ratios:
                    # Get average convergence rate
                    avg_rate = np.mean(ratios)
                    sparsity_values.append(result['sparsity'])
                    conv_rates.append(avg_rate)
        
        # Sort by sparsity
        if sparsity_values:
            sort_idx = np.argsort(sparsity_values)
            sparsity_values = [sparsity_values[i] for i in sort_idx]
            conv_rates = [conv_rates[i] for i in sort_idx]
            
            plt.plot(sparsity_values, conv_rates, 'o-', linewidth=2, 
                     label=f"{rule}")
    
    # Add line for discount factor
    plt.axhline(y=0.9, color='r', linestyle='--', label='Discount Factor (γ=0.9)')
    
    plt.xlabel('Sparsity Level')
    plt.ylabel('Average Convergence Rate (e_k+1/e_k)')
    plt.title('Effect of Sparsity on Convergence Rate (n_states=200)')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\nComprehensive MDP Algorithm Benchmark")
    print("=" * 80)
    
    # Choose which benchmarks to run
    run_main_benchmark = True
    run_sparsity_analysis = True
    
    if run_main_benchmark:
        # Run the standard comprehensive benchmark
        benchmarker = run_comprehensive_benchmark()
        
        # Create comparison tables
        tables = create_comparison_tables(benchmarker)
        print_tables(tables)
        
        # Create visualizations
        create_visualizations(benchmarker)
        
        # Print analysis summary
        print_analysis_summary(benchmarker)
    
    if run_sparsity_analysis:
        # Run the sparsity benchmark
        print("\n" + "=" * 80)
        print("Running Sparsity Impact Analysis")
        print("=" * 80 + "\n")
        
        # Run the sparsity benchmark and get results
        sparsity_df, sparsity_results = run_sparsity_benchmark()
        
        # Create sparsity visualizations
        create_sparsity_visualizations(sparsity_df, sparsity_results)
        
        # Print sparsity analysis summary
        print("\nSparsity Analysis Summary")
        print("=" * 80)
        
        # Group by rule and sparsity level
        for rule in sparsity_df['param_rule'].unique():
            print(f"\nPerformance of {rule} across sparsity levels:")
            
            rule_df = sparsity_df[sparsity_df['param_rule'] == rule]
            
            # Filter for only the largest state size for clearer comparison
            large_state_df = rule_df[rule_df['n_states'] == 200]
            
            # Sort by sparsity and show iterations
            pivot = large_state_df.pivot_table(
                index='sparsity',
                values=['iterations', 'runtime'],
                aggfunc='mean'
            ).sort_index()
            
            print(pivot)
            
        # Compare all algorithms at high sparsity
        high_sparsity_df = sparsity_df[sparsity_df['sparsity'] >= 0.9]
        high_sparsity_pivot = high_sparsity_df.pivot_table(
            index='param_rule',
            values=['iterations', 'runtime'],
            aggfunc='mean'
        ).sort_values('iterations')
        
        print("\nAlgorithm performance on highly sparse MDPs (sparsity >= 0.9):")
        print(high_sparsity_pivot)
    
    print("\nAll benchmarks completed!")
    print("=" * 80)