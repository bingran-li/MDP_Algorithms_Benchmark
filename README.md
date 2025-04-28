# MDP Library

A modular and efficient library for generating, solving, and benchmarking Markov Decision Processes (MDPs).

## Features

- **Modular Design**: Separation of problem generation, algorithm logic, and metrics
- **Reproducibility**: Random seeds for consistent results and experiment recreation
- **Efficiency**: Vectorized operations and optimized implementations
- **Extensibility**: Easily add new algorithms or problem types
- **Benchmarking Tools**: Comprehensive comparison of algorithm performance
- **Visualization**: Plot convergence and performance metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mdp-lib.git
cd mdp-lib

# Install requirements
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from mdp_lib import MDPGenerator, AlgorithmFactory

# Create MDP generator with seed for reproducibility
mdp_gen = MDPGenerator(seed=42)

# Generate an MDP
mdp = mdp_gen.generate_mdp(
    n_states=10,
    m_actions=4,
    mdp_type='stochastic',
    discount_factor=0.9
)

# Create algorithm
policy_iter = AlgorithmFactory.create('policy_iteration', seed=42)

# Solve MDP
result = policy_iter.solve(mdp, rule='standard')

# Access solution
policy = result['policy']
values = result['values']
print(f"Iterations: {result['iterations']}")
print(f"Runtime: {result['runtime']} seconds")
```

### Algorithm Benchmarking

```python
from mdp_lib import Benchmarker

# Create benchmarker with seed
benchmarker = Benchmarker(seed=42)

# Prepare multiple MDPs
mdp_specs = [
    (10, 'stochastic'),  # (n_states, mdp_type)
    (50, 'stochastic'),
    (100, 'stochastic'),
    (10, 'deterministic')
]
benchmarker.prepare_mdps(mdp_specs)

# Define algorithms and parameters to test
algorithms = ['policy_iteration', 'value_iteration']
param_grid = {
    'policy_iteration': {'rule': ['standard', 'modified']},
    'value_iteration': {'rule': ['standard', 'gauss-seidel']}
}

# Run benchmark
benchmarker.benchmark_algorithms(algorithms, param_grid)

# Generate comparison table
comparison = benchmarker.create_comparison_table()
print(comparison)

# Plot performance comparison
benchmarker.plot_performance_comparison(metric='runtime_seconds', by='n_states')
```

## Project Structure

- `mdp_lib/` - Core library code
  - `generator.py` - MDP generation module
  - `algorithms.py` - Algorithm implementations
  - `benchmarking.py` - Benchmarking and comparison tools
- `code/` - Example code and notebooks
- `results/` - Saved benchmark results

## Available Algorithms

- **Policy Iteration**
  - Standard (direct linear system solution)
  - Modified (iterative policy evaluation)
  
- **Value Iteration**
  - Standard
  - Gauss-Seidel
  - Prioritized Sweeping
  
- **Simplex Method**
  - Basic implementation

## Requirements

- NumPy
- SciPy
- Pandas
- Matplotlib
- tqdm

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 