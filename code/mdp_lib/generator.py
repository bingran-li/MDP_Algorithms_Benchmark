import numpy as np
from scipy.stats import dirichlet
import json
import os
from datetime import datetime


class MDPGenerator:
    """
    Class for generating Markov Decision Process (MDP) problems
    with reproducible random seeding.
    """
    def __init__(self, seed=None):
        """
        Initialize the MDP generator
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        self.mdp_types = ['deterministic', 'stochastic']
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
    
    def generate_mdp(self, n_states, m_actions, mdp_type='stochastic', 
                     discount_factor=0.9, reward_range=(0, 1), 
                     sparsity=0.0, terminal_states=None):
        """
        Generate a random MDP problem
        
        Args:
            n_states (int): Number of states
            m_actions (int): Number of actions
            mdp_type (str): 'deterministic' or 'stochastic'
            discount_factor (float): Discount factor gamma (0 to 1)
            reward_range (tuple): Range of rewards (min, max)
            sparsity (float): Probability of zero transition (0 to 1)
            terminal_states (list): List of terminal states
            
        Returns:
            dict: MDP problem specification
        """
        if mdp_type not in self.mdp_types:
            raise ValueError(f"MDP type must be one of {self.mdp_types}")
        
        # Initialize MDP components
        transitions = np.zeros((n_states, m_actions, n_states))
        rewards = np.zeros((n_states, m_actions))
        
        # Generate transitions and rewards
        for s in range(n_states):
            for a in range(m_actions):
                if mdp_type == 'deterministic':
                    # For deterministic, only one next state gets probability 1
                    if terminal_states and s in terminal_states:
                        transitions[s, a, s] = 1.0  # Self-loop for terminal states
                    else:
                        next_state = np.random.randint(n_states)
                        transitions[s, a, next_state] = 1.0
                else:  # stochastic
                    if terminal_states and s in terminal_states:
                        transitions[s, a, s] = 1.0  # Self-loop for terminal states
                    else:
                        # Apply sparsity - make some transitions have zero probability
                        if sparsity > 0:
                            mask = np.random.random(n_states) > sparsity
                            # Ensure at least one non-zero transition
                            if not np.any(mask):
                                mask[np.random.randint(n_states)] = True
                            
                            # Get indices of non-zero transitions
                            valid_states = np.where(mask)[0]
                            # Generate probabilities only for valid states
                            probs = dirichlet.rvs([1] * len(valid_states))[0]
                            # Assign probabilities to valid states
                            transitions[s, a, valid_states] = probs
                        else:
                            transitions[s, a] = dirichlet.rvs([1]*n_states)[0]
                
                # Generate rewards
                min_reward, max_reward = reward_range
                rewards[s, a] = min_reward + np.random.rand() * (max_reward - min_reward)
        
        # Create MDP specification
        mdp = {
            'transitions': transitions,
            'rewards': rewards,
            'discount_factor': discount_factor,
            'n_states': n_states,
            'n_actions': m_actions,
            'type': mdp_type,
            'terminal_states': terminal_states,
            'seed': self.seed,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return mdp
    
    def generate_mdp_batch(self, sizes, action_counts=None, mdp_types=None, 
                           discount_factor=0.9, reward_range=(0, 1), sparsity=0.0):
        """
        Generate a batch of MDPs with different configurations
        
        Args:
            sizes (list): List of state space sizes
            action_counts (list, optional): List of action space sizes. Defaults to [4].
            mdp_types (list, optional): List of MDP types. Defaults to ['stochastic'].
            discount_factor (float): Discount factor for all MDPs
            reward_range (tuple): Reward range for all MDPs
            sparsity (float): Sparsity for all MDPs
            
        Returns:
            list: List of generated MDPs
        """
        if action_counts is None:
            action_counts = [4]
        if mdp_types is None:
            mdp_types = ['stochastic']
            
        mdps = []
        
        for n_states in sizes:
            for m_actions in action_counts:
                for mdp_type in mdp_types:
                    # Use a different seed for each MDP but derived from the base seed
                    # This ensures reproducibility while still having different MDPs
                    if self.seed is not None:
                        mdp_seed = self.seed + len(mdps)
                        np.random.seed(mdp_seed)
                    else:
                        mdp_seed = None
                        
                    mdp = self.generate_mdp(
                        n_states=n_states,
                        m_actions=m_actions,
                        mdp_type=mdp_type,
                        discount_factor=discount_factor,
                        reward_range=reward_range,
                        sparsity=sparsity
                    )
                    
                    # Add metadata for tracking
                    mdp['metadata'] = {
                        'batch_idx': len(mdps),
                        'mdp_seed': mdp_seed
                    }
                    
                    mdps.append(mdp)
        
        return mdps
    
    def save_mdp(self, mdp, filename):
        """
        Save MDP to a file
        
        Args:
            mdp (dict): MDP specification
            filename (str): Path to save the MDP
        """
        # Convert numpy arrays to lists for JSON serialization
        mdp_copy = mdp.copy()
        mdp_copy['transitions'] = mdp['transitions'].tolist()
        mdp_copy['rewards'] = mdp['rewards'].tolist()
        
        with open(filename, 'w') as f:
            json.dump(mdp_copy, f, indent=2)
    
    def load_mdp(self, filename):
        """
        Load MDP from a file
        
        Args:
            filename (str): Path to the MDP file
            
        Returns:
            dict: MDP specification
        """
        with open(filename, 'r') as f:
            mdp = json.load(f)
        
        # Convert lists back to numpy arrays
        mdp['transitions'] = np.array(mdp['transitions'])
        mdp['rewards'] = np.array(mdp['rewards'])
        
        return mdp 