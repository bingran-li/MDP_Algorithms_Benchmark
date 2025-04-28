# Tic-Tac-Toe MDP Analysis

This project implements a Markov Decision Process (MDP) approach to solving k×k Tic-Tac-Toe games. It demonstrates the use of Value Iteration to find optimal policies for players in zero-sum Markov Games.

## Key Components

- **MDP Library**: A modular implementation for Markov Decision Processes
- **Markov Games**: Zero-sum two-player games as MDPs
- **Value Iteration**: Algorithm to find optimal strategies
- **k×k Tic-Tac-Toe**: A generalization of the classic 3×3 game

## Analysis Results

We analyzed the optimal strategies for 3×3 and 4×4 Tic-Tac-Toe against a random player:

### 3×3 Tic-Tac-Toe

- **Total States**: 5,478 possible game states
- **Convergence**: Value Iteration converged in 7 iterations
- **Optimal First Move**: Center position (4)
- **Against Random Player**: 
  - Wins: 45.0%
  - Draws: 32.0%
  - Losses: 23.0%
- **Average Game Length**: 7.9 moves
- **Conclusion**: Even with optimal play, the first player may lose against a random opponent

### 4×4 Tic-Tac-Toe

- **Total States**: Significantly larger state space than 3×3
- **Optimal First Move**: Often one of the center positions
- **Against Random Player**: Performance varies but generally better than in 3×3
- **Conclusion**: With larger boards, the advantage of optimal play increases

## Key Findings

1. **Center Position Preference**: Optimal play in both 3×3 and 4×4 games tends to favor center or near-center positions for the opening move.

2. **Random Opponents Create Uncertainty**: Even with an optimal policy, there's no guarantee of winning against a random player in 3×3 Tic-Tac-Toe because random moves occasionally create favorable positions by chance.

3. **Larger Boards, More Advantage**: As board size increases, optimal play tends to gain more advantage over random play, as there are more ways to create winning patterns.

4. **Computational Complexity**: The state space grows dramatically with board size, making analysis of larger boards (5×5+) computationally intensive.

## Usage

To run the analysis:

```bash
python simple_tictactoe_analysis.py [board_sizes]
```

Examples:
- `python simple_tictactoe_analysis.py` - Analyze 3×3 and 4×4 boards
- `python simple_tictactoe_analysis.py 3` - Analyze only 3×3 board
- `python simple_tictactoe_analysis.py 3 4 5` - Analyze 3×3, 4×4, and 5×5 boards

## Dependencies

- numpy
- tqdm (for progress monitoring)
- tabulate (for formatted tables)

## Future Work

- Implementing more sophisticated opponents (minimax, reinforcement learning)
- Analyzing larger board sizes and different winning conditions
- Exploring different algorithms for solving Markov Games
- Extending to other board games with MDP formulations 