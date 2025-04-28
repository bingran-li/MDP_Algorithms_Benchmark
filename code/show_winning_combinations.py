#!/usr/bin/env python
"""
Display winning combinations for k×k Tic-Tac-Toe boards
"""
import sys
import os
from tabulate import tabulate

def generate_winning_combinations(board_size):
    """Generate all possible winning combinations for a k×k board"""
    k = board_size
    combinations = []
    
    # Rows
    for row in range(k):
        combinations.append([row * k + col for col in range(k)])
    
    # Columns
    for col in range(k):
        combinations.append([row * k + col for row in range(k)])
    
    # Main diagonal
    combinations.append([i * k + i for i in range(k)])
    
    # Other diagonal
    combinations.append([i * k + (k - 1 - i) for i in range(k)])
    
    return combinations

def print_board_layout(board_size):
    """Print the board layout with position numbers"""
    k = board_size
    
    print(f"\n{k}×{k} Board Layout:")
    divider = "-" * (4 * k + 1)
    
    print(divider)
    for row in range(k):
        cells = []
        for col in range(k):
            idx = row * k + col
            cells.append(f"{idx:2d}")
        print("| " + " | ".join(cells) + " |")
        print(divider)

def visualize_winning_combinations(board_size):
    """Visualize each winning combination on a board"""
    k = board_size
    combinations = generate_winning_combinations(board_size)
    
    print(f"\nWinning Combinations for {k}×{k} Board:")
    
    for idx, combo in enumerate(combinations):
        # Create empty board
        board = [' ' for _ in range(k*k)]
        
        # Mark winning positions
        for pos in combo:
            board[pos] = 'X'
            
        # Determine the type of winning combination
        if idx < k:
            combo_type = f"Row {idx}"
        elif idx < 2*k:
            combo_type = f"Column {idx-k}"
        elif idx == 2*k:
            combo_type = "Main diagonal"
        else:
            combo_type = "Other diagonal"
            
        print(f"\nCombination {idx+1}: {combo_type}")
        
        # Print board with the winning combination
        divider = "-" * (4 * k + 1)
        print(divider)
        for row in range(k):
            cells = []
            for col in range(k):
                idx = row * k + col
                cells.append(f"{board[idx]}")
            print("| " + " | ".join(cells) + " |")
            print(divider)

def analyze_board_size(k):
    """Analyze a specific board size"""
    print(f"\n{'='*50}")
    print(f"ANALYZING {k}×{k} TIC-TAC-TOE BOARD")
    print(f"{'='*50}")
    
    # Display board layout
    print_board_layout(k)
    
    # Generate and count winning combinations
    combinations = generate_winning_combinations(k)
    
    # Display statistics
    print(f"\nSTATISTICS:")
    print(f"- Board size: {k}×{k}")
    print(f"- Total cells: {k*k}")
    print(f"- Winning combinations: {len(combinations)}")
    print(f"  - Rows: {k}")
    print(f"  - Columns: {k}")
    print(f"  - Diagonals: 2")
    
    # Ask if user wants to see all combinations
    if k <= 5:
        visualize_winning_combinations(k)
    else:
        print("\nBoard too large to visualize all combinations!")

if __name__ == "__main__":
    # Process command line arguments
    if len(sys.argv) > 1:
        # User specified board sizes
        for arg in sys.argv[1:]:
            try:
                k = int(arg)
                if k < 3:
                    print(f"Board size {k} is too small (minimum 3)")
                    continue
                analyze_board_size(k)
            except ValueError:
                print(f"Invalid board size: {arg}")
    else:
        # Default: analyze 3×3, 4×4, and 5×5 boards
        for k in [3, 4, 5]:
            analyze_board_size(k) 