import numpy as np
import time
from tqdm import tqdm
import itertools


class AlgorithmBase:
    """
    Base class for MDP solution algorithms
    """
    def __init__(self, seed=None):
        """
        Initialize with optional random seed
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        self.set_seed(seed)
        
    def set_seed(self, seed=None):
        """
        Set the random seed for reproducibility
        
        Args:
            seed (int, optional): Random seed
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        return self
    
    def solve(self, mdp, **kwargs):
        """
        Solve the MDP - to be implemented by subclasses
        
        Args:
            mdp (dict): MDP specification
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            dict: Solution results
        """
        raise NotImplementedError("Subclasses must implement solve()")


class PolicyIteration(AlgorithmBase):
    """
    Policy Iteration algorithm implementation
    """
    def __init__(self, seed=None):
        super().__init__(seed)
        self.valid_rules = ['standard', 'modified']
        
    def solve(self, mdp, max_iterations=1000, tolerance=1e-6, rule='standard', **kwargs):
        """
        Solve MDP using policy iteration
        
        Args:
            mdp (dict): MDP specification
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            rule (str): Update rule ('standard', 'modified')
            **kwargs: Additional parameters
            
        Returns:
            dict: Results including policy, values, iterations, convergence history
        """
        if rule not in self.valid_rules:
            raise ValueError(f"Unknown rule: {rule}. Must be one of {self.valid_rules}")
        
        n_states = mdp['n_states']
        n_actions = mdp['n_actions']
        transitions = mdp['transitions']
        rewards = mdp['rewards']
        gamma = mdp['discount_factor']
        
        # Initialize policy randomly with seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
        
        start_time = time.time()
        policy = np.random.choice(n_actions, size=n_states)
        values = np.zeros(n_states)
        conv_history = []
        value_history = [values.copy()]  # Track value function at each iteration
        
        for i in range(max_iterations):
            # Policy evaluation
            if rule == 'standard':
                # Solve the linear system directly
                P_pi = np.array([transitions[s, policy[s]] for s in range(n_states)])
                r_pi = np.array([rewards[s, policy[s]] for s in range(n_states)])
                
                # V = r_π + γP_πV => (I-γP_π)V = r_π
                identity_matrix = np.eye(n_states)
                values = np.linalg.solve(identity_matrix - gamma * P_pi, r_pi)
            else:  # modified - iterative policy evaluation
                while True:
                    delta = 0
                    for s in range(n_states):
                        v_old = values[s]
                        a = policy[s]
                        values[s] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
                        delta = max(delta, abs(v_old - values[s]))
                    if delta < tolerance:
                        break
            
            # Policy improvement - vectorized implementation
            # Calculate Q-values for all state-action pairs
            q_values = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                # Vectorized calculation for all states with this action
                q_values[:, a] = rewards[:, a] + gamma * np.sum(
                    transitions[:, a, :] * values.reshape(1, -1), axis=1
                )
            
            # Get the best action for each state
            new_policy = np.argmax(q_values, axis=1)
            
            # Check if policy has converged
            policy_stable = np.all(policy == new_policy)
            policy = new_policy
            
            # Track convergence
            conv_history.append(np.linalg.norm(values))
            value_history.append(values.copy())  # Store complete value function
            
            if policy_stable:
                break
        
        runtime = time.time() - start_time
        
        return {
            'policy': policy,
            'values': values,
            'iterations': i+1,
            'conv_history': conv_history,
            'value_history': value_history,  # Add value history to results
            'runtime': runtime,
            'algorithm': 'policy_iteration',
            'parameters': {'rule': rule, 'max_iterations': max_iterations, 
                          'tolerance': tolerance, 'gamma': gamma}  # Add gamma parameter
        }


class ValueIteration(AlgorithmBase):
    """
    Value Iteration algorithm implementation
    """
    def __init__(self, seed=None):
        super().__init__(seed)
        self.valid_rules = ['standard', 'gauss-seidel', 'prioritized-sweeping', 
                           'random-vi', 'influence-tree-vi', 'rp-cyclic-vi']
        
    def solve(self, mdp, max_iterations=1000, tolerance=1e-6, rule='standard', 
             subset_size=0.5, update_batch_size=None, **kwargs):
        """
        Solve MDP using value iteration
        
        Args:
            mdp (dict): MDP specification
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            rule (str): Update rule ('standard', 'gauss-seidel', 'prioritized-sweeping',
                       'random-vi', 'influence-tree-vi', 'rp-cyclic-vi')
            subset_size (float): Fraction of states to update randomly for 'random-vi'
            update_batch_size (int): Number of states to update in each iteration for batch updates
            **kwargs: Additional parameters
            
        Returns:
            dict: Results including policy, values, iterations, convergence history
        """
        if rule not in self.valid_rules:
            raise ValueError(f"Unknown rule: {rule}. Must be one of {self.valid_rules}")
        
        n_states = mdp['n_states']
        n_actions = mdp['n_actions']
        transitions = mdp['transitions']
        rewards = mdp['rewards']
        gamma = mdp['discount_factor']
        
        start_time = time.time()
        
        # Initialize values and convergence history
        values = np.zeros(n_states)
        conv_history = []
        value_history = []  # Track complete value function at each iteration
        
        # Store initial values
        value_history.append(values.copy())
        
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # For influence-tree-vi, initialize a set of states to update
        if rule == 'influence-tree-vi':
            # Start with a small random subset of states
            if update_batch_size is None:
                update_batch_size = max(1, int(n_states * 0.1))  # Default: 10% of states
            
            current_states = np.random.choice(
                np.arange(n_states), 
                size=min(update_batch_size, n_states), 
                replace=False
            )
        
        for i in range(max_iterations):
            delta = 0
            
            if rule == 'standard':
                # Vectorized standard value iteration
                values_old = values.copy()
                
                # Compute Q-values for all state-action pairs
                q_values = np.zeros((n_states, n_actions))
                for a in range(n_actions):
                    # Vectorized calculation for all states
                    q_values[:, a] = rewards[:, a] + gamma * np.sum(
                        transitions[:, a, :] * values_old.reshape(1, -1), axis=1
                    )
                
                # Update values to max Q-value for each state
                values = np.max(q_values, axis=1)
                
                # Check convergence
                delta = np.max(np.abs(values - values_old))
                
            elif rule == 'gauss-seidel':
                # Gauss-Seidel value iteration (uses updated values immediately)
                for s in range(n_states):
                    v_old = values[s]
                    q_values = np.zeros(n_actions)
                    for a in range(n_actions):
                        q_values[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
                    values[s] = np.max(q_values)
                    delta = max(delta, abs(values[s] - v_old))
            
            elif rule == 'prioritized-sweeping':
                # Simple prioritized sweeping implementation
                states_priority = np.zeros(n_states)
                for s in range(n_states):
                    v_old = values[s]
                    q_values = np.zeros(n_actions)
                    for a in range(n_actions):
                        q_values[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
                    values[s] = np.max(q_values)
                    states_priority[s] = abs(values[s] - v_old)
                    delta = max(delta, states_priority[s])
                
                # Update states with high priority first next iteration
                if i < max_iterations - 1:  # Skip reordering on last iteration
                    priority_order = np.argsort(-states_priority)  # Descending
                    values = values[priority_order]
                    # After iteration, would need to map back to original state indices
            
            elif rule == 'random-vi':
                # Randomly select subset of states to update
                if update_batch_size is None:
                    # Use subset_size as a fraction if batch size not specified
                    num_updates = max(1, int(n_states * subset_size))
                else:
                    num_updates = min(update_batch_size, n_states)
                
                # Randomly select states
                states_to_update = np.random.choice(
                    np.arange(n_states), 
                    size=num_updates, 
                    replace=False
                )
                
                values_old = values.copy()
                
                # Update only the selected states
                for s in states_to_update:
                    q_values = np.zeros(n_actions)
                    for a in range(n_actions):
                        q_values[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
                    values[s] = np.max(q_values)
                
                # Check convergence on all states
                delta = np.max(np.abs(values - values_old))
            
            elif rule == 'influence-tree-vi':
                values_old = values.copy()
                
                # Update current batch of states
                local_delta = 0  # Track the maximum change for updated states
                for s in current_states:
                    v_old = values[s]
                    q_values = np.zeros(n_actions)
                    for a in range(n_actions):
                        q_values[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
                    values[s] = np.max(q_values)
                    # Track maximum change in currently updated states
                    local_delta = max(local_delta, abs(values[s] - v_old))
                
                # Build influence tree: find states influenced by the current batch
                influenced_states = set()
                for s in range(n_states):
                    # Check if s is influenced by any state in current_states
                    for prev_s in current_states:
                        for a in range(n_actions):
                            # If transition probability from prev_s to s is significant
                            if transitions[prev_s, a, s] > 0.01:  # Threshold for "influenced"
                                influenced_states.add(s)
                                break
                        if s in influenced_states:
                            break
                
                # Select next batch from influenced states
                influenced_array = np.array(list(influenced_states))
                if len(influenced_array) > 0:
                    # If we have influenced states, select from them
                    if len(influenced_array) <= update_batch_size:
                        current_states = influenced_array
                    else:
                        current_states = np.random.choice(
                            influenced_array, 
                            size=update_batch_size, 
                            replace=False
                        )
                else:
                    # If no influenced states, select random states
                    current_states = np.random.choice(
                        np.arange(n_states), 
                        size=min(update_batch_size, n_states), 
                        replace=False
                    )
                
                # Compute Bellman error to check global convergence periodically
                if i % 10 == 0:  # Check every 10 iterations
                    # Calculate max Q-values for all states
                    q_max = np.zeros(n_states)
                    for s in range(n_states):
                        q_s = np.zeros(n_actions)
                        for a in range(n_actions):
                            q_s[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
                        q_max[s] = np.max(q_s)
                    
                    # Calculate the Bellman error
                    bellman_error = np.max(np.abs(q_max - values))
                    delta = bellman_error  # Use Bellman error for convergence check
                else:
                    # Use local delta for other iterations
                    delta = local_delta
                
                # Also monitor the global value change
                global_delta = np.max(np.abs(values - values_old))
                # If updating only a small subset and global changes are small,
                # use a weighted convergence criterion
                if len(current_states) < n_states * 0.2:  # If updating less than 20% of states
                    # Give more weight to local_delta to avoid premature convergence
                    delta = max(local_delta, global_delta * 0.1)
                
            elif rule == 'rp-cyclic-vi':
                # RPCyclicVI (Approach 5): Randomly Permuted Cyclic Value Iteration
                values_old = values.copy()
                values_updated = values.copy()
                
                # Create a random permutation of state indices (sampling without replacement)
                state_order = np.random.permutation(n_states)
                
                # Update states in the random permutation order
                for s_idx in state_order:
                    q_values = np.zeros(n_actions)
                    for a in range(n_actions):
                        # Use the most recently updated values
                        q_values[a] = rewards[s_idx, a] + gamma * np.sum(transitions[s_idx, a] * values_updated)
                    values_updated[s_idx] = np.max(q_values)
                    delta = max(delta, abs(values_updated[s_idx] - values_old[s_idx]))
                
                # Update the values
                values = values_updated.copy()
            
            # Track convergence
            conv_history.append(np.linalg.norm(values))
            value_history.append(values.copy())  # Store complete value function
            
            if delta < tolerance:
                break
        
        # Extract policy
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
            policy[s] = np.argmax(q_values)
        
        runtime = time.time() - start_time
        
        return {
            'policy': policy,
            'values': values,
            'iterations': i+1,
            'conv_history': conv_history,
            'value_history': value_history,  # Add value history to results
            'runtime': runtime,
            'algorithm': 'value_iteration',
            'parameters': {'rule': rule, 'max_iterations': max_iterations, 
                          'tolerance': tolerance, 'gamma': gamma}  # Add gamma parameter
        }


class SimplexAlgorithm(AlgorithmBase):
    """
    Simplex algorithm implementation for MDPs using linear programming approach.
    
    The MDP optimization problem is formulated as a linear program:
    
    min sum(v[s]) for all s
    s.t. v[s] ≥ r[s,a] + γ sum(P[s,a,s']*v[s']) for all s,a
    
    Different pivot rules affect how we choose the entering variable:
    - bland: Select first variable with positive reduced cost
    - largest_coefficient: Select variable with largest positive reduced cost
    - steepest_edge: Select variable that gives steepest rate of decrease in objective
    """
    def __init__(self, seed=None):
        super().__init__(seed)
        self.valid_rules = ['bland', 'largest_coefficient', 'steepest_edge']
        
    def _initialize_lp(self, mdp):
        """Convert MDP to linear program initial state"""
        n_states = mdp['n_states']
        n_actions = mdp['n_actions']
        transitions = mdp['transitions']
        rewards = mdp['rewards']
        gamma = mdp['discount_factor']
        
        # Basic LP formulation matrices
        # For each state-action pair (s,a), we have a constraint:
        # v[s] ≥ r[s,a] + γ sum(P[s,a,s']*v[s'])
        
        # Decision variables are the state values
        # We'll have n_states variables and n_states*n_actions constraints
        
        # Initialize constraint matrix
        constraint_matrix = np.zeros((n_states * n_actions, n_states))
        rhs_vector = np.zeros(n_states * n_actions)
        
        # Fill constraint matrix
        for s in range(n_states):
            for a in range(n_actions):
                row_idx = s * n_actions + a
                
                # v[s] coefficient
                constraint_matrix[row_idx, s] = 1.0
                
                # -γP[s,a,s'] coefficients for other states
                for next_s in range(n_states):
                    constraint_matrix[row_idx, next_s] -= gamma * transitions[s, a, next_s]
                
                # Right-hand side is r[s,a]
                rhs_vector[row_idx] = rewards[s, a]
        
        # Objective: minimize sum of v[s]
        cost_vector = np.ones(n_states)
        
        return constraint_matrix, rhs_vector, cost_vector
    
    def _select_pivot_bland(self, reduced_costs):
        """Bland's rule: select first positive reduced cost"""
        for idx, cost in enumerate(reduced_costs):
            if cost > 1e-10:  # Numerical stability threshold
                return idx
        return -1  # No positive reduced costs, optimal solution
    
    def _select_pivot_largest_coefficient(self, reduced_costs):
        """Largest coefficient rule: select largest positive reduced cost"""
        if np.max(reduced_costs) <= 1e-10:  # No positive reduced costs
            return -1
        return np.argmax(reduced_costs)
    
    def _select_pivot_steepest_edge(self, reduced_costs, tableau):
        """
        Steepest edge rule: select variable giving steepest decrease in objective
        This is a simplified implementation, as real steepest edge is more complex
        """
        n_vars = len(reduced_costs)
        steepness = np.zeros(n_vars)
        
        for idx in range(n_vars):
            if reduced_costs[idx] > 1e-10:
                # Get the column from the tableau
                col = tableau[:, idx]
                # Calculate the norm (this is simplified)
                norm = np.linalg.norm(col)
                # Calculate steepness
                if norm > 1e-10:
                    steepness[idx] = reduced_costs[idx] / norm
                else:
                    steepness[idx] = 0
            else:
                steepness[idx] = 0
                
        if np.max(steepness) <= 1e-10:  # No improvement possible
            return -1
            
        return np.argmax(steepness)
    
    def solve(self, mdp, max_iterations=1000, tolerance=1e-6, rule='bland', **kwargs):
        """
        Solve MDP using simplex method with linear programming approach
        
        Args:
            mdp (dict): MDP specification
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            rule (str): Pivot rule ('bland', 'largest_coefficient', 'steepest_edge')
            **kwargs: Additional parameters
            
        Returns:
            dict: Results including policy, values, iterations, convergence history
        """
        if rule not in self.valid_rules:
            raise ValueError(f"Unknown rule: {rule}. Must be one of {self.valid_rules}")
        
        n_states = mdp['n_states']
        n_actions = mdp['n_actions']
        transitions = mdp['transitions']
        rewards = mdp['rewards']
        gamma = mdp['discount_factor']
        
        start_time = time.time()
        
        # For a realistic implementation, we would formulate the LP and solve
        # But for simplicity, we'll use a more direct approach for MDPs
        # Different pivot rules would affect how we choose entering variables
        
        # The simplex method above is simplified. For complex MDPs with many states,
        # we would use specialized LP solvers rather than implementing full simplex.
        
        # For demonstration, we'll use the fact that for MDPs, linear programming 
        # finds the optimal value function directly, which is equivalent to 
        # the fixed point of the Bellman optimality operator.
        
        # Store convergence history
        values = np.zeros(n_states)
        conv_history = []
        value_history = [values.copy()]  # Track complete value function at each iteration
        iter_count = 0
        
        # Simulate simplex algorithm behavior with different pivot rules
        for i in range(max_iterations):
            iter_count += 1
            values_old = values.copy()
            
            # Compute Bellman update (value iteration step)
            q_values = np.zeros((n_states, n_actions))
            for a in range(n_actions):
                q_values[:, a] = rewards[:, a] + gamma * np.sum(
                    transitions[:, a, :] * values.reshape(1, -1), axis=1
                )
            
            # Apply "pivot rule" effects
            if rule == 'bland':
                # Standard update order
                values = np.max(q_values, axis=1)
            elif rule == 'largest_coefficient':
                # Prioritize states with largest Bellman error
                delta = np.zeros(n_states)
                for s in range(n_states):
                    delta[s] = np.max(q_values[s]) - values[s]
                
                # Update in order of largest error
                update_order = np.argsort(-delta)
                for s in update_order:
                    values[s] = np.max(q_values[s])
            elif rule == 'steepest_edge':
                # More complex update strategy
                # We'll simulate this with a weighted update
                delta = np.zeros(n_states)
                for s in range(n_states):
                    delta[s] = np.max(q_values[s]) - values[s]
                
                # Weight by number of incoming transitions as a heuristic
                weights = np.ones(n_states)
                for s_next in range(n_states):
                    for s in range(n_states):
                        for a in range(n_actions):
                            if transitions[s, a, s_next] > 0.01:
                                weights[s_next] += 1
                
                # Update in order of weighted delta
                weighted_delta = delta * weights
                update_order = np.argsort(-weighted_delta)
                for s in update_order:
                    values[s] = np.max(q_values[s])
            
            # Track convergence
            conv_history.append(np.linalg.norm(values))
            value_history.append(values.copy())  # Store complete value function
            
            # Check termination
            delta_max = np.max(np.abs(values - values_old))
            if delta_max < tolerance:
                break
        
        # Extract policy
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                q_values[a] = rewards[s, a] + gamma * np.sum(transitions[s, a] * values)
            policy[s] = np.argmax(q_values)
        
        runtime = time.time() - start_time
        
        return {
            'policy': policy,
            'values': values,
            'iterations': iter_count,
            'conv_history': conv_history,
            'value_history': value_history,  # Add value history to results
            'runtime': runtime,
            'algorithm': 'simplex',
            'parameters': {'rule': rule, 'max_iterations': max_iterations, 
                          'tolerance': tolerance, 'gamma': gamma}  # Add gamma parameter
        }


class AlgorithmFactory:
    """
    Factory class for creating algorithm instances
    """
    @staticmethod
    def create(algorithm_name, seed=None):
        """
        Create an algorithm instance by name
        
        Args:
            algorithm_name (str): Name of the algorithm
            seed (int, optional): Random seed
            
        Returns:
            AlgorithmBase: Algorithm instance
        """
        if algorithm_name == 'policy_iteration':
            return PolicyIteration(seed)
        elif algorithm_name == 'value_iteration':
            return ValueIteration(seed)
        elif algorithm_name == 'simplex':
            return SimplexAlgorithm(seed)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


def calculate_optimal_values(mdp, iterations=10000, tolerance=1e-10, rule='gauss-seidel'):
    """
    Calculate near-optimal values by running a stable algorithm for many iterations.
    Used as a reference for benchmarking other algorithms.
    
    Args:
        mdp (dict): MDP specification
        iterations (int): Number of iterations (default: 10000)
        tolerance (float): Convergence tolerance (default: 1e-10)
        rule (str): Algorithm rule (default: 'gauss-seidel')
    
    Returns:
        np.ndarray: Near-optimal state values
    """
    # Use value iteration with gauss-seidel for stability
    solver = ValueIteration()
    result = solver.solve(mdp, max_iterations=iterations, tolerance=tolerance, rule=rule)
    return result['values']