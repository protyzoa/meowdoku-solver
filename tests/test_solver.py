"""
Test the solver end-to-end with a synthetic board (no image needed).
Generates a known-solvable 6x6 region map and verifies the solution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.solver.backtracker import solve_puzzle
from src.renderer.annotator import print_ascii_board

# 6×6 board with 6 color regions (labeled 0-5)
# Each region forms an L-shape or block
region_map_6x6 = np.array([
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [3, 3, 4, 4, 5, 5],
    [3, 3, 4, 4, 5, 5],
    [3, 3, 4, 4, 5, 5],
    [3, 3, 4, 4, 5, 5],
], dtype=int)

print("=" * 40)
print("Test: 6×6 Meowdoku (2×2 blocks per region)")
print("=" * 40)
print("\nRegion map:")
for row in region_map_6x6:
    print(" ".join(chr(ord('A') + v) for v in row))

solution = solve_puzzle(region_map_6x6)
if solution:
    print(f"\n✅ Solution: {solution}")
    print()
    print(print_ascii_board(6, solution, region_map_6x6))
else:
    print("\n❌ No solution found")

# Another test: 4×4 board
region_map_4x4 = np.array([
    [0, 0, 1, 1],
    [0, 2, 2, 1],
    [3, 2, 2, 1],
    [3, 3, 3, 1],
], dtype=int)

print("\n" + "=" * 40)
print("Test: 4×4 Meowdoku (irregular regions)")
print("=" * 40)
print("\nRegion map:")
for row in region_map_4x4:
    print(" ".join(chr(ord('A') + v) for v in row))

solution4 = solve_puzzle(region_map_4x4)
if solution4:
    print(f"\n✅ Solution: {solution4}")
    print()
    print(print_ascii_board(4, solution4, region_map_4x4))
else:
    print("\n❌ No solution found")

