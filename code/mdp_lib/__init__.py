"""
MDP Library - A modular implementation of MDP algorithms

This library implements various algorithms for solving Markov Decision Processes (MDPs),
including policy iteration, value iteration, and simplex methods.
"""

from .generator import MDPGenerator
from .algorithms import (
    AlgorithmBase, 
    AlgorithmFactory,
    PolicyIteration,
    ValueIteration,
    SimplexAlgorithm
)
from .benchmarking import Benchmarker

# Import the Markov Game Processes module
from .markov_games import (
    MarkovGameProcess,
    TicTacToeGame,
    MGPValueIteration,
    play_tic_tac_toe_game,
    solve_tic_tac_toe_example
)

__all__ = [
    'MDPGenerator',
    'AlgorithmBase',
    'AlgorithmFactory',
    'PolicyIteration',
    'ValueIteration',
    'SimplexAlgorithm',
    'Benchmarker',
    'MarkovGameProcess',
    'TicTacToeGame',
    'MGPValueIteration',
    'play_tic_tac_toe_game',
    'solve_tic_tac_toe_example'
] 