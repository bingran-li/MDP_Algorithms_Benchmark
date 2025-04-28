#%%
import numpy as np
from typing import Tuple, List, Optional

class MDP:
    def __init__(
        self,
        P: np.ndarray,  # Transition matrix: [action, state, next_state]
        c: np.ndarray,  # Cost matrix: [action, state]
        gamma: float = 0.9,
    ):
        self.P = P
        self.c = c
        self.gamma = gamma
        self.n_actions, self.n_states, _ = P.shape

    def value_iteration(
        self, 
        y0: Optional[np.ndarray] = None, 
        max_iter: int = 1000, 
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Approach 1: Standard Value Iteration.
        Args:
            y0: Initial value vector (shape: [n_states])
            max_iter: Maximum iterations
            tol: Convergence tolerance
        Returns:
            y: Optimal value vector
            history: List of ||y_k - y*||_∞ at each iteration
        """
        y = np.zeros(self.n_states) if y0 is None else y0.copy()
        history = []
        
        for _ in range(max_iter):
            y_prev = y.copy()
            Q = self.c + self.gamma * (self.P @ y)  # [action, state]
            y = np.min(Q, axis=0)  # Update all states
            
            error = np.max(np.abs(y - y_prev))
            history.append(error)
            if error < tol:
                break
                
        return y, history

    def randomized_vi(
        self,
        y0: Optional[np.ndarray] = None,
        batch_size: int = 1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Approach 2: Randomized Value Iteration (updates a random subset of states each iteration).
        Args:
            y0: Initial value vector
            batch_size: Number of states to update per iteration (|B_k|)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            seed: Random seed
        Returns:
            y: Optimal value vector
            history: List of max changes in updated states per iteration
        """
        np.random.seed(seed)
        y = np.zeros(self.n_states) if y0 is None else y0.copy()
        history = []
        
        for _ in range(max_iter):
            y_prev = y.copy()
            Bk = np.random.choice(self.n_states, size=batch_size, replace=False)
            
            Q = self.c + self.gamma * (self.P @ y)  # [action, state]
            y[Bk] = np.min(Q[:, Bk], axis=0)  # Only update states in Bk
            
            error = np.max(np.abs(y[Bk] - y_prev[Bk]))  # Track only updated states
            history.append(error)
            if error < tol:
                break
                
        return y, history

def test_mdp():
    # Example MDP: 3 states, 2 actions
    P = np.array([
        # Action 0 (shape [state, next_state])
        [[0.7, 0.2, 0.1],  # From state 0
         [0.0, 0.8, 0.2],   # From state 1
         [0.5, 0.5, 0.0]],  # From state 2
        
        # Action 1
        [[0.1, 0.8, 0.1],
         [0.3, 0.7, 0.0],
         [0.0, 0.0, 1.0]]
    ])
    
    c = np.array([
        [1.0, 2.0, 0.5],  # Action 0 costs
        [0.5, 1.0, 3.0]    # Action 1 costs
    ])
    
    mdp = MDP(P, c, gamma=0.9)
    
    # Approach 1: Value Iteration
    y_vi, hist_vi = mdp.value_iteration()
    print("Optimal values (VI):", y_vi)
    
    # Approach 2: Randomized VI
    y_rvi, hist_rvi = mdp.randomized_vi(batch_size=2)
    print("Optimal values (Randomized VI):", y_rvi)

    return (y_vi, hist_vi), (y_rvi, hist_rvi)

if __name__ == "__main__":
    (y_vi, hist_vi), (y_rvi, hist_rvi) = test_mdp()