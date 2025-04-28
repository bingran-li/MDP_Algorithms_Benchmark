import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from tqdm import tqdm
import json
import os
from datetime import datetime

from .generator import MDPGenerator
from .algorithms import AlgorithmFactory


class Benchmarker:
    """
    Class for benchmarking MDP algorithms
    """
    def __init__(self, seed=None):
        """
        Initialize benchmarker with optional seed
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        self.seed = seed
        self.results = []
        self.mdps = []
        self.df_results = None
        
    def set_seed(self, seed):
        """
        Set random seed
        
        Args:
            seed (int): Random seed
        """
        self.seed = seed
        np.random.seed(seed)
        return self
    
    def prepare_mdps(self, mdp_specs=None, generator=None):
        """
        Prepare MDPs for benchmarking
        
        Args:
            mdp_specs: List of MDP specifications or tuples (n_states, mdp_type)
            generator: MDP generator to use
            
        Returns:
            list: List of prepared MDPs
        """
        if mdp_specs is None:
            mdp_specs = [
                (10, 'stochastic'), 
                (50, 'stochastic'),
                (100, 'stochastic')
            ]
            
        if generator is None:
            generator = MDPGenerator(seed=self.seed)
        
        processed_mdps = []
        
        for idx, spec in enumerate(mdp_specs):
            if isinstance(spec, tuple):
                n_states, mdp_type = spec
                # Use a different seed for each MDP
                mdp_seed = None if self.seed is None else self.seed + idx
                
                # Generate MDP with default parameters
                mdp = generator.set_seed(mdp_seed).generate_mdp(
                    n_states=n_states,
                    m_actions=4,
                    mdp_type=mdp_type
                )
                
                # Add metadata for tracking
                mdp['metadata'] = {
                    'spec_idx': idx,
                    'spec_type': 'tuple',
                    'spec': spec,
                    'seed': mdp_seed
                }
                
                processed_mdps.append(mdp)
            else:
                # Assume it's already an MDP dict
                mdp = spec.copy()
                
                # Add metadata if it doesn't exist
                if 'metadata' not in mdp:
                    mdp['metadata'] = {
                        'spec_idx': idx,
                        'spec_type': 'dict',
                        'seed': self.seed
                    }
                    
                processed_mdps.append(mdp)
        
        self.mdps = processed_mdps
        return processed_mdps
    
    def run_algorithm(self, mdp, algorithm_name, params=None):
        """
        Run a single algorithm on an MDP
        
        Args:
            mdp (dict): MDP specification
            algorithm_name (str): Name of the algorithm
            params (dict, optional): Algorithm parameters
            
        Returns:
            dict: Algorithm results
        """
        if params is None:
            params = {}
            
        # Create algorithm with derived seed
        alg_seed = None if self.seed is None else self.seed + len(self.results)
        algorithm = AlgorithmFactory.create(algorithm_name, seed=alg_seed)
        
        # Run algorithm
        start_time = time.time()
        result = algorithm.solve(mdp, **params)
        runtime = time.time() - start_time
        
        # Add metadata
        if 'runtime' not in result:
            result['runtime'] = runtime
            
        if 'algorithm' not in result:
            result['algorithm'] = algorithm_name
            
        if 'parameters' not in result:
            result['parameters'] = params
            
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
            
        if 'seed' not in result:
            result['seed'] = alg_seed
            
        # Add MDP metadata
        result['mdp_metadata'] = mdp.get('metadata', {})
        result['n_states'] = mdp['n_states']
        result['n_actions'] = mdp['n_actions']
        result['mdp_type'] = mdp['type']
        
        return result
    
    def benchmark_algorithms(self, algorithms=None, param_grid=None, n_runs=1):
        """
        Benchmark multiple algorithms across all stored MDPs
        
        Args:
            algorithms: List of algorithm names to test
            param_grid: Dictionary of parameter grids to search
                        {alg_name: {param_name: [param_values]}}
            n_runs: Number of runs for each configuration (for averaging)
        
        Returns:
            list: Benchmark results
        """
        if not self.mdps:
            raise ValueError("No MDPs available. Call prepare_mdps first.")
            
        if algorithms is None:
            algorithms = ['policy_iteration', 'value_iteration', 'simplex']
        
        if param_grid is None:
            param_grid = {
                'policy_iteration': {'rule': ['standard', 'modified']},
                'value_iteration': {'rule': ['standard', 'gauss-seidel']},
                'simplex': {'rule': ['bland', 'largest_coefficient', 'steepest_edge']}
            }
        
        results = []
        
        # For each MDP and algorithm combination
        for mdp_idx, mdp in enumerate(tqdm(self.mdps, desc='MDPs')):
            for alg in algorithms:
                # Get parameter grid for this algorithm
                alg_params = param_grid.get(alg, {})
                param_names = list(alg_params.keys())
                param_values = list(alg_params.values())
                
                # Generate all parameter combinations
                param_combinations = [
                    dict(zip(param_names, combo))
                    for combo in itertools.product(*param_values)
                ]
                
                # If no parameters specified, use empty dict
                if not param_combinations:
                    param_combinations = [{}]
                
                # Test each parameter combination
                for params in param_combinations:
                    # Run n_runs times for averaging
                    for run in range(n_runs):
                        # Adjust seed for each run
                        run_seed = None
                        if self.seed is not None:
                            run_seed = self.seed + mdp_idx + len(results) + run
                            np.random.seed(run_seed)
                            
                        # Run algorithm with these parameters
                        result = self.run_algorithm(mdp, alg, params)
                        
                        # Add run information
                        result['run_idx'] = run
                        result['run_seed'] = run_seed
                        
                        # Store results
                        results.append(result)
        
        self.results = results
        return results
    
    def results_to_dataframe(self):
        """
        Convert benchmark results to a pandas DataFrame
        
        Returns:
            pd.DataFrame: Benchmark results as DataFrame
        """
        if not self.results:
            raise ValueError("No benchmark results available")
            
        # Extract relevant data from results
        rows = []
        
        for r in self.results:
            # Create a basic row with essential metrics
            row = {
                'algorithm': r['algorithm'],
                'n_states': r['n_states'],
                'n_actions': r['n_actions'],
                'mdp_type': r['mdp_type'],
                'iterations': r['iterations'],
                'runtime': r['runtime'],  # Use consistent 'runtime' column name
                'mean_value': np.mean(r['values']),
                'max_value': np.max(r['values']),
                'min_value': np.min(r['values']),
                'seed': r.get('seed', None),
                'run_idx': r.get('run_idx', 0),
                'run_seed': r.get('run_seed', None),
                'mdp_idx': r.get('mdp_metadata', {}).get('spec_idx', None),
                'mdp_seed': r.get('mdp_metadata', {}).get('seed', None),
                'timestamp': r.get('timestamp', None)
            }
            
            # Add algorithm parameters
            for k, v in r.get('parameters', {}).items():
                param_key = f'param_{k}'
                row[param_key] = v
                
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by important columns for better organization
        sort_cols = ['algorithm', 'param_rule', 'n_states', 'mdp_type', 'run_idx']
        sort_cols = [col for col in sort_cols if col in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols)
            
        # Reset index for cleaner display
        df = df.reset_index(drop=True)
        
        self.df_results = df
        return df

    def save_results(self, filename):
        """
        Save benchmark results to file
        
        Args:
            filename (str): Path to save results
        """
        if not self.results:
            raise ValueError("No benchmark results available")
            
        # Convert results to serializable format
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_results = [make_serializable(r) for r in self.results]
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'seed': self.seed,
                'results': serializable_results
            }, f, indent=2)
    
    def load_results(self, filename):
        """
        Load benchmark results from file
        
        Args:
            filename (str): Path to results file
            
        Returns:
            list: Benchmark results
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Convert lists back to numpy arrays
        results = data['results']
        
        for r in results:
            # Convert common arrays back to numpy
            for key in ['policy', 'values', 'conv_history', 'value_history']:
                if key in r and isinstance(r[key], list):
                    r[key] = np.array(r[key])
                
        self.results = results
        self.seed = data.get('seed', None)
        
        return results
    
    def create_comparison_table(self, group_by=None, metrics=None, 
                               sort_by=None, ascending=True):
        """
        Create a comparison table of algorithm performance
        
        Args:
            group_by: Columns to group by
            metrics: Metrics to include in comparison
            sort_by: Column to sort by
            ascending: Sort order
            
        Returns:
            pd.DataFrame: Comparison table
        """
        if self.df_results is None:
            self.results_to_dataframe()
            
        if group_by is None:
            group_by = ['algorithm', 'param_rule', 'n_states', 'mdp_type']
            
        # Use consistent column naming and check what's available
        # Determine runtime column name
        runtime_col = 'runtime_seconds' if 'runtime_seconds' in self.df_results.columns else 'runtime'
        
        if metrics is None:
            metrics = [
                'iterations', runtime_col, 'mean_value', 
                'max_value', 'min_value'
            ]
            
        # Ensure all group_by columns exist
        existing_columns = self.df_results.columns.tolist()
        group_by = [col for col in group_by if col in existing_columns]
        metrics = [col for col in metrics if col in existing_columns]
        
        # Create comparison table with various aggregation functions
        agg_funcs = {
            'iterations': ['mean', 'min', 'max', 'std', 'count'],
            'mean_value': ['mean', 'std'],
            'max_value': ['mean'],
            'min_value': ['mean']
        }
        
        # Add runtime column with correct name
        agg_funcs[runtime_col] = ['mean', 'min', 'max', 'std']
        
        # Filter to only existing metrics
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in metrics}
        
        # Create multi-level aggregation
        comparison = self.df_results.groupby(group_by).agg(agg_funcs)
        
        # Flatten column hierarchy for better display
        comparison.columns = ['_'.join(col).strip() for col in comparison.columns.values]
        
        # Reset index for better display
        comparison = comparison.reset_index()
        
        # Sort by specified column or default to algorithm and n_states
        if sort_by is None:
            if 'algorithm' in group_by and 'n_states' in group_by:
                sort_by = ['algorithm', 'n_states']
            elif 'algorithm' in group_by:
                sort_by = ['algorithm']
            elif 'n_states' in group_by:
                sort_by = ['n_states']
            else:
                sort_by = group_by[0] if group_by else None
                
        if sort_by:
            if isinstance(sort_by, list):
                # Ensure all sort columns exist
                sort_by = [col for col in sort_by if col in comparison.columns]
                if sort_by:  # Only sort if there are valid columns
                    comparison = comparison.sort_values(by=sort_by, ascending=ascending)
            else:
                # Check if the sort column exists
                if sort_by in comparison.columns:
                    comparison = comparison.sort_values(by=sort_by, ascending=ascending)
            
        return comparison
    
    def create_pivot_table(self, values=None, index=None, 
                           columns=None, aggfunc='mean'):
        """
        Create a pivot table for easier comparison
        
        Args:
            values: Value column(s) to aggregate
            index: Column(s) to use as index
            columns: Column(s) to use as columns
            aggfunc: Aggregation function
            
        Returns:
            pd.DataFrame: Pivot table
        """
        if self.df_results is None:
            self.results_to_dataframe()
            
        # Determine runtime column name
        runtime_col = 'runtime_seconds' if 'runtime_seconds' in self.df_results.columns else 'runtime'
            
        if values is None:
            values = ['iterations', runtime_col, 'mean_value']
            
        if index is None:
            index = ['n_states', 'mdp_type']
            
        if columns is None:
            columns = ['algorithm', 'param_rule']
            
        # Ensure all columns exist
        existing_columns = self.df_results.columns.tolist()
        values = [col for col in values if col in existing_columns]
        index = [col for col in index if col in existing_columns]
        columns = [col for col in columns if col in existing_columns]
        
        # Create pivot table
        pivot = pd.pivot_table(
            self.df_results,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc
        )
        
        # Rename columns for better display
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = ['_'.join(str(c) for c in col).strip() 
                            for col in pivot.columns.values]
        
        # Calculate performance ratios if possible
        try:
            # Try to add performance ratio between algorithms using the correct runtime column name
            pi_col = f"{runtime_col}_policy_iteration_standard"
            vi_col = f"{runtime_col}_value_iteration_standard"
            simplex_col = f"{runtime_col}_simplex_bland"
            
            if pi_col in pivot.columns and vi_col in pivot.columns:
                pivot['speedup_pi_vs_vi'] = pivot[vi_col] / pivot[pi_col]
                
            if pi_col in pivot.columns and simplex_col in pivot.columns:
                pivot['speedup_pi_vs_simplex'] = pivot[simplex_col] / pivot[pi_col]
        except Exception:
            # Silently ignore if calculation fails
            pass
            
        return pivot
    
    def plot_convergence(self, subset=None, figsize=(12, 8), use_optimal_values=True, save_path=None):
        """
        Plot convergence curves for different algorithms
        
        Args:
            subset: Subset of results indices to plot
            figsize: Figure size
            use_optimal_values: Whether to use pre-calculated optimal values if available
            save_path: Path to save the figure (default: results/convergence_plot.png)
        """
        if not self.results:
            raise ValueError("No benchmark results available")
            
        # Use subset or all results
        plot_results = self.results
        if subset is not None:
            if isinstance(subset, list):
                plot_results = [self.results[i] for i in subset]
            else:
                # Filter based on condition function
                plot_results = [r for i, r in enumerate(self.results) 
                               if subset(r, i)]
        
        plt.figure(figsize=figsize)
        
        # Group by MDP size
        mdp_sizes = sorted(list(set(r['n_states'] for r in plot_results)))
        
        for size in mdp_sizes:
            plt.subplot(len(mdp_sizes), 1, mdp_sizes.index(size) + 1)
            size_results = [r for r in plot_results if r['n_states'] == size]
            
            # Group by MDP type to ensure we compare within the same MDP type
            mdp_types = set(r['mdp_type'] for r in size_results)
            
            for mdp_type in mdp_types:
                type_results = [r for r in size_results if r['mdp_type'] == mdp_type]
                
                # Group results by MDP instance
                mdp_groups = {}
                for result in type_results:
                    mdp_idx = result.get('mdp_metadata', {}).get('spec_idx', 0)
                    if mdp_idx not in mdp_groups:
                        mdp_groups[mdp_idx] = []
                    mdp_groups[mdp_idx].append(result)
                
                # For each MDP instance, plot convergence compared to the optimal solution
                for mdp_idx, mdp_results in mdp_groups.items():
                    # First, try to find optimal values that were calculated during benchmarking
                    # This would be the case if run_comprehensive_benchmark added optimal_values
                    optimal_values = None
                    
                    if use_optimal_values:
                        # Check if any result has 'optimal_values'
                        for r in mdp_results:
                            if 'optimal_values' in r and r['optimal_values'] is not None:
                                optimal_values = r['optimal_values']
                                break
                                
                    # If no optimal values were found and use_optimal_values is True, log a warning
                    if use_optimal_values and optimal_values is None:
                        print(f"Warning: No pre-calculated optimal values found for MDP {mdp_idx}. Using best available values.")
                                
                    # If we couldn't find pre-calculated optimal values, use the best available as a fallback
                    if optimal_values is None:
                        max_iterations = 0
                        for r in mdp_results:
                            iterations = r.get('iterations', 0)
                            if iterations > max_iterations:
                                max_iterations = iterations
                                # Assuming values is the state values array
                                optimal_values = r.get('values', None)
                    
                    # Skip if we couldn't find values
                    if optimal_values is None:
                        continue
                        
                    # Plot error convergence for each algorithm on this MDP
                    for result in mdp_results:
                        alg = result['algorithm']
                        params = result['parameters']
                        params_str = '_'.join(f"{k}:{v}" for k, v in params.items())
                        label = f"{alg} ({params_str}) - {mdp_type} MDP #{mdp_idx}"
                        
                        # Extract convergence history
                        conv_history = result.get('conv_history', [])
                        
                        # Compute errors if convergence history exists
                        if len(conv_history) > 0 and 'value_history' in result:
                            value_history = result['value_history']
                            errors = []
                            
                            for values in value_history:
                                # Compute absolute error from optimal values
                                error = np.max(np.abs(values - optimal_values))
                                errors.append(error)
                            
                            # Plot error convergence (log scale for better visualization)
                            plt.semilogy(errors, label=label)
                        elif len(conv_history) > 0:
                            # If value_history not available, use the original convergence history
                            plt.semilogy(conv_history, label=f"{label} (original metric)")
            
            plt.title(f"MDP Size: {size} states")
            plt.xlabel('Iterations')
            plt.ylabel('Error (|V - V*|)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path is None:
            # Create a default path in the results directory
            save_path = 'results/convergence_plot'
            # Add MDP size info to filename if focusing on specific size
            if len(mdp_sizes) == 1:
                save_path += f'_size{mdp_sizes[0]}'
            # Add timestamp to avoid overwriting
            save_path += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
        
        plt.show()

    def plot_performance_comparison(self, metric='runtime', by='n_states', 
                                   figsize=(12, 8), save_path=None):
        """
        Compare algorithm performance across different MDP configurations
        
        Args:
            metric: Metric to compare ('iterations', 'runtime', 'runtime_seconds')
            by: Variable to compare across ('n_states', 'n_actions')
            figsize: Figure size
            save_path: Path to save the figure (default: auto-generated)
        """
        if self.df_results is None:
            self.results_to_dataframe()
            
        # Make sure we're using a valid metric
        if metric not in self.df_results.columns:
            # Try to use runtime with the correct column name
            if metric in ['runtime', 'runtime_seconds']:
                metric = 'runtime_seconds' if 'runtime_seconds' in self.df_results.columns else 'runtime'
            else:
                raise ValueError(f"Metric {metric} not found in results")
            
        plt.figure(figsize=figsize)
        
        # Identify all parameter columns
        param_columns = [col for col in self.df_results.columns if col.startswith('param_')]
        
        # Group by algorithm and existing parameters
        group_columns = ['algorithm'] + param_columns
        
        # Get unique algorithm-parameter combinations
        alg_param_groups = self.df_results[group_columns].drop_duplicates()
        
        x_values = sorted(self.df_results[by].unique())
        
        # Plot one line for each algorithm-parameter combination
        for _, row in alg_param_groups.iterrows():
            # Create label from algorithm and parameters
            alg = row['algorithm']
            params_str = '_'.join(f"{col.replace('param_', '')}:{row[col]}" 
                                for col in param_columns if pd.notna(row[col]))
            label = f"{alg} ({params_str})" if params_str else alg
            
            # Filter data for this combination
            mask = (self.df_results['algorithm'] == alg)
            for col in param_columns:
                if pd.notna(row[col]):
                    mask = mask & (self.df_results[col] == row[col])
            
            subset = self.df_results[mask]
            
            # Get data points for each x value
            y_values = []
            for x in x_values:
                x_subset = subset[subset[by] == x]
                if not x_subset.empty:
                    y_values.append(x_subset[metric].mean())
                else:
                    y_values.append(np.nan)
            
            plt.plot(x_values, y_values, 'o-', label=label)
        
        plt.xlabel(by.capitalize().replace('_', ' '))
        plt.ylabel(metric.capitalize().replace('_', ' '))
        plt.title(f'Algorithm Performance: {metric.capitalize().replace("_", " ")} vs {by.capitalize().replace("_", " ")}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path is None:
            # Create a default path in the results directory
            metric_name = metric.lower().replace('_', '')
            by_name = by.lower().replace('_', '')
            save_path = f'results/performance_{metric_name}_vs_{by_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison plot to {save_path}")
        
        plt.show()
        return plt
    
    def plot_boxplot_comparison(self, metric='runtime', group_by=None,
                               figsize=(12, 8), save_path=None):
        """
        Create boxplot comparison of algorithm performance
        
        Args:
            metric: Metric to compare
            group_by: How to group algorithms
            figsize: Figure size
            save_path: Path to save the figure (default: auto-generated)
        """
        if self.df_results is None:
            self.results_to_dataframe()
            
        # Make sure we're using a valid metric
        if metric not in self.df_results.columns:
            # Try to use runtime with the correct column name
            if metric in ['runtime', 'runtime_seconds']:
                metric = 'runtime_seconds' if 'runtime_seconds' in self.df_results.columns else 'runtime'
            else:
                raise ValueError(f"Metric {metric} not found in results")
            
        if group_by is None:
            group_by = ['algorithm']
            
        # Check if all group_by columns exist
        existing_columns = self.df_results.columns.tolist()
        group_by = [col for col in group_by if col in existing_columns]
        
        if not group_by:
            raise ValueError("No valid group_by columns found")
        
        plt.figure(figsize=figsize)
        
        # Create shorter labels for the groups
        labels = []
        
        # Get unique combinations of group_by columns
        unique_groups = self.df_results[group_by].drop_duplicates()
        
        # Create mapping from original group names to shorter labels
        group_to_label = {}
        for i, row in unique_groups.iterrows():
            # Extract group values
            group_vals = [row[col] for col in group_by]
            
            # Construct original group name (this will be used in _group column)
            original_group = '_'.join([str(val) for val in group_vals])
            
            # Create shorter label
            if len(group_by) == 1 and group_by[0] == 'algorithm':
                # Just use algorithm name
                label = str(row['algorithm'])
            elif 'algorithm' in group_by and 'param_rule' in group_by:
                # Use algorithm + rule format
                alg_idx = group_by.index('algorithm')
                rule_idx = group_by.index('param_rule')
                label = f"{group_vals[alg_idx]}\n({group_vals[rule_idx]})"
            else:
                # Generic format
                label = '\n'.join([f"{col}:{val}" for col, val in zip(group_by, group_vals)])
            
            group_to_label[original_group] = label
            labels.append(label)
        
        # Create a temporary column for grouping
        self.df_results['_group'] = self.df_results[group_by].astype(str).agg('_'.join, axis=1)
        
        # Create box plot
        bp = self.df_results.boxplot(
            column=metric,
            by='_group',
            grid=True,
            return_type='dict',
            rot=0  # No rotation initially, we'll use our custom labels
        )
        
        # Replace x-tick labels with our shorter ones
        plt.gca().set_xticklabels(labels)
        
        plt.title(f'Performance Comparison: {metric.capitalize().replace("_", " ")}')
        plt.suptitle('')  # Remove default title
        plt.xlabel('Algorithm Configuration')
        plt.ylabel(metric.replace('_', ' ').capitalize())
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path is None:
            # Create a default path in the results directory
            metric_name = metric.lower().replace('_', '')
            group_str = '_'.join([col.lower().replace('_', '') for col in group_by])
            save_path = f'results/boxplot_{metric_name}_by_{group_str}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved boxplot comparison to {save_path}")
        
        plt.show()
        
        # Remove temporary column
        self.df_results.drop(columns=['_group'], inplace=True)
        
        return bp

    def plot_improved_convergence(self, subset=None, figsize=(12, 8), use_bellman_error=True, save_path=None):
        """
        Plot improved convergence curves for different algorithms
        
        Args:
            subset: Subset of results indices to plot
            figsize: Figure size
            use_bellman_error: If True, compute and use Bellman error instead of 
                              difference from 'optimal' values
            save_path: Path to save the figure (default: auto-generated)
        """
        if not self.results:
            raise ValueError("No benchmark results available")
            
        # Use subset or all results
        plot_results = self.results
        if subset is not None:
            if isinstance(subset, list):
                plot_results = [self.results[i] for i in subset]
            else:
                # Filter based on condition function
                plot_results = [r for i, r in enumerate(self.results) 
                               if subset(r, i)]
        
        plt.figure(figsize=figsize)
        
        # Group by MDP size
        mdp_sizes = sorted(list(set(r['n_states'] for r in plot_results)))
        
        for size in mdp_sizes:
            plt.subplot(len(mdp_sizes), 1, mdp_sizes.index(size) + 1)
            size_results = [r for r in plot_results if r['n_states'] == size]
            
            # Find the best value as "optimal" for each MDP size
            # Group by MDP type to ensure we compare within the same MDP type
            mdp_types = set(r['mdp_type'] for r in size_results)
            
            for mdp_type in mdp_types:
                type_results = [r for r in size_results if r['mdp_type'] == mdp_type]
                
                # Group results by MDP instance
                mdp_groups = {}
                for result in type_results:
                    mdp_idx = result.get('mdp_metadata', {}).get('spec_idx', 0)
                    if mdp_idx not in mdp_groups:
                        mdp_groups[mdp_idx] = []
                    mdp_groups[mdp_idx].append(result)
                
                # For each MDP instance, plot convergence
                for mdp_idx, mdp_results in mdp_groups.items():
                    # Check if any result has optimal values precalculated
                    optimal_values = None
                    for r in mdp_results:
                        if 'optimal_values' in r and r['optimal_values'] is not None:
                            optimal_values = r['optimal_values']
                            break
                    
                    # Find the corresponding MDP
                    mdp = None
                    for r in mdp_results:
                        mdp_metadata = r.get('mdp_metadata', {})
                        mdp_spec_idx = mdp_metadata.get('spec_idx', None)
                        if mdp_spec_idx is not None and mdp_spec_idx < len(self.mdps):
                            mdp = self.mdps[mdp_spec_idx]
                            break
                    
                    if mdp is None:
                        print(f"Warning: Could not find MDP for idx {mdp_idx}")
                        continue
                    
                    # Get MDP parameters for Bellman error calculation
                    transitions = mdp.get('transitions')
                    rewards = mdp.get('rewards')
                    gamma = mdp.get('discount_factor', 0.9)
                    
                    # If no pre-calculated optimal values, use best available
                    if optimal_values is None:
                        max_iterations = 0
                        for r in mdp_results:
                            iterations = r.get('iterations', 0)
                            if iterations > max_iterations:
                                max_iterations = iterations
                                optimal_values = r.get('values', None)
                    
                    # Skip if we couldn't find values
                    if optimal_values is None:
                        continue
                        
                    # Plot error convergence for each algorithm on this MDP
                    for result in mdp_results:
                        alg = result['algorithm']
                        params = result['parameters']
                        params_str = '_'.join(f"{k}:{v}" for k, v in params.items())
                        label = f"{alg} ({params_str}) - {mdp_type}"
                        
                        # Extract value history
                        value_history = result.get('value_history', [])
                        
                        # Compute errors if value history exists
                        if len(value_history) > 0:
                            errors = []
                            
                            for values in value_history:
                                if use_bellman_error and transitions is not None and rewards is not None:
                                    # Compute Bellman error (more accurate measure of convergence)
                                    bellman_error = 0
                                    for s in range(len(values)):
                                        # Current value
                                        v = values[s]
                                        
                                        # Compute max Q-value (Bellman update)
                                        q_values = []
                                        for a in range(rewards.shape[1]):  # n_actions
                                            q = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
                                            q_values.append(q)
                                        
                                        # Bellman error is |v - max_a Q(s,a)|
                                        bellman_error = max(bellman_error, abs(v - max(q_values)))
                                    
                                    errors.append(bellman_error)
                                else:
                                    # Compute absolute error from optimal values
                                    error = np.max(np.abs(values - optimal_values))
                                    errors.append(error)
                            
                            # Plot error convergence (log scale for better visualization)
                            plt.semilogy(errors, label=label)
            
            plt.title(f"MDP Size: {size} states")
            plt.xlabel('Iterations')
            if use_bellman_error:
                plt.ylabel('Bellman Error (max|v(s) - max_a Q(s,a)|)')
            else:
                plt.ylabel('Error (max|V - V*|)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path is None:
            # Create a default path in the results directory
            save_path = 'results/improved_convergence'
            # Add MDP size info to filename if focusing on specific size
            if len(mdp_sizes) == 1:
                save_path += f'_size{mdp_sizes[0]}'
            # Add indicator of error type
            if use_bellman_error:
                save_path += '_bellman'
            else:
                save_path += '_optimal'
            # Add timestamp to avoid overwriting
            save_path += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved improved convergence plot to {save_path}")
        
        plt.show()
    
    def compute_optimal_solution(self, mdp):
        """
        Compute a near-optimal solution by running a stable algorithm for many iterations
        
        Args:
            mdp (dict): MDP to solve
            
        Returns:
            np.ndarray: Near-optimal state values
        """
        # Use value iteration with Gauss-Seidel (known to be stable)
        algorithm = AlgorithmFactory.create('value_iteration')
        
        # Run for 10000 iterations with tight tolerance to ensure convergence
        result = algorithm.solve(
            mdp, 
            max_iterations=10000,  # Very large number of iterations 
            tolerance=1e-10,       # Very tight tolerance
            rule='gauss-seidel'    # Stable update rule
        )
        
        return result['values']
    
    def run_benchmark_with_optimal_reference(self, mdp, algorithm_name, params=None):
        """
        Run a benchmark with comparison to optimal values
        
        Args:
            mdp (dict): MDP to solve
            algorithm_name (str): Algorithm to use
            params (dict): Parameters for the algorithm
            
        Returns:
            dict: Benchmark results with optimal value comparison
        """
        from .algorithms import calculate_optimal_values
        
        # Calculate optimal values first as reference
        print(f"Calculating optimal solution using reference algorithm...")
        optimal_values = calculate_optimal_values(mdp, iterations=10000, tolerance=1e-10)
        
        # Run the actual algorithm
        if params is None:
            params = {}
        
        print(f"Running {algorithm_name} with parameters {params}...")
        result = self.run_algorithm(mdp, algorithm_name, params)
        
        # Calculate error from optimal solution
        error_from_optimal = np.max(np.abs(result['values'] - optimal_values))
        mean_error = np.mean(np.abs(result['values'] - optimal_values))
        
        # Add to results
        result['optimal_values'] = optimal_values
        result['error_from_optimal'] = error_from_optimal
        result['mean_error_from_optimal'] = mean_error
        
        print(f"Completed in {result['runtime']:.4f} seconds with {result['iterations']} iterations")
        print(f"Max error from optimal: {error_from_optimal:.6f}, Mean error: {mean_error:.6f}")
        
        return result