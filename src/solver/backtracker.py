"""
Meowdoku CSP Solver — Backtracking with forward checking.

Constraints:
  1. Exactly one cat per row.
  2. Exactly one cat per column.
  3. Exactly one cat per color region.
  4. No two cats may be adjacent (including diagonally).

Optionally:
  5. Exactly one cat per 3×3 block (if board_size is divisible by 3).
"""

import numpy as np
from typing import List, Optional, Tuple, Set


class MeowdokuSolver:
    """
    Solves a Meowdoku puzzle given a region map.

    region_map: 2D list/array of shape (N, N)
                Each cell contains an integer region ID (0-indexed).
    """

    def __init__(self, region_map: np.ndarray, use_3x3_blocks: bool = False):
        self.region_map = np.array(region_map, dtype=int)
        self.n = self.region_map.shape[0]
        self.n_regions = int(self.region_map.max()) + 1
        self.use_3x3_blocks = use_3x3_blocks and (self.n % 3 == 0)

        # Track which rows/cols/regions/blocks are already used
        self.row_used: Set[int] = set()
        self.col_used: Set[int] = set()
        self.region_used: Set[int] = set()
        self.block_used: Set[int] = set()

        # Solution grid: -1 = empty, 1 = cat
        self.board = np.full((self.n, self.n), -1, dtype=int)

        # Candidate domain for each row: which columns are still possible
        self.domains: List[Set[int]] = [set(range(self.n)) for _ in range(self.n)]

    # ─────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """
        Run the solver.

        Returns:
            List of (row, col) positions for each cat (sorted by row),
            or None if no solution found.
        """
        if not self._validate_region_map():
            raise ValueError(
                f"Region map has {self.n_regions} regions but board is {self.n}×{self.n}. "
                f"Meowdoku requires exactly N regions for an N×N board."
            )

        self._apply_initial_propagation()
        success = self._backtrack(row=0)
        if not success:
            return None

        positions = []
        for r in range(self.n):
            for c in range(self.n):
                if self.board[r, c] == 1:
                    positions.append((r, c))
        return sorted(positions)

    # ─────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────

    def _validate_region_map(self) -> bool:
        return self.n_regions == self.n

    def _block_id(self, r: int, c: int) -> int:
        block_size = 3
        return (r // block_size) * (self.n // block_size) + (c // block_size)

    def _is_valid(self, r: int, c: int) -> bool:
        """Check all constraints for placing a cat at (r, c)."""
        # 1. Row & column
        if r in self.row_used or c in self.col_used:
            return False

        # 2. Region
        region = int(self.region_map[r, c])
        if region in self.region_used:
            return False

        # 3. No adjacency (8 neighbors)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    if self.board[nr, nc] == 1:
                        return False

        # 4. 3×3 block (optional)
        if self.use_3x3_blocks:
            if self._block_id(r, c) in self.block_used:
                return False

        return True

    def _place(self, r: int, c: int):
        self.board[r, c] = 1
        self.row_used.add(r)
        self.col_used.add(c)
        self.region_used.add(int(self.region_map[r, c]))
        if self.use_3x3_blocks:
            self.block_used.add(self._block_id(r, c))

    def _remove(self, r: int, c: int):
        self.board[r, c] = -1
        self.row_used.discard(r)
        self.col_used.discard(c)
        self.region_used.discard(int(self.region_map[r, c]))
        if self.use_3x3_blocks:
            self.block_used.discard(self._block_id(r, c))

    def _apply_initial_propagation(self):
        """
        Simple propagation: if a region has only one cell,
        that cell must contain a cat.
        """
        for region_id in range(self.n_regions):
            cells = list(zip(*np.where(self.region_map == region_id)))
            if len(cells) == 1:
                r, c = cells[0]
                # Remove column from domains of other rows
                for other_r in range(self.n):
                    if other_r != r:
                        self.domains[other_r].discard(c)

    def _forward_check(self, r: int, c: int) -> bool:
        """
        After placing a cat at (r, c), check that no future row
        has an empty domain. Returns False if any domain becomes empty.
        """
        placed_col = c
        placed_region = int(self.region_map[r, c])

        for future_r in range(r + 1, self.n):
            valid_cols = set()
            for fc in self.domains[future_r]:
                if (fc not in self.col_used and
                        int(self.region_map[future_r, fc]) not in self.region_used and
                        not self._adjacent_to_any_cat(future_r, fc)):
                    valid_cols.add(fc)
            if not valid_cols:
                return False
        return True

    def _adjacent_to_any_cat(self, r: int, c: int) -> bool:
        """Returns True if (r, c) is adjacent to any already-placed cat."""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.n and 0 <= nc < self.n:
                    if self.board[nr, nc] == 1:
                        return True
        return False

    def _backtrack(self, row: int) -> bool:
        """Recursive backtracking row by row."""
        if row == self.n:
            # All rows placed — verify we have exactly N cats
            return len(self.row_used) == self.n

        # Sort candidate columns: prefer cells whose region has fewer remaining
        # options (MRV heuristic)
        candidates = sorted(
            self.domains[row],
            key=lambda c: self._count_region_options(row, c)
        )

        for c in candidates:
            if self._is_valid(row, c):
                self._place(row, c)

                # Forward checking
                if self._forward_check(row, c):
                    if self._backtrack(row + 1):
                        return True

                self._remove(row, c)

        return False

    def _count_region_options(self, r: int, c: int) -> int:
        """
        Count how many valid placements remain for the region of (r, c).
        Used as MRV heuristic — prefer more constrained regions first.
        """
        region = int(self.region_map[r, c])
        cells = list(zip(*np.where(self.region_map == region)))
        count = 0
        for cr, cc in cells:
            if cr not in self.row_used and cc not in self.col_used:
                count += 1
        return count


def solve_puzzle(region_map: np.ndarray, use_3x3_blocks: bool = False) -> Optional[List[Tuple[int, int]]]:
    """
    Convenience wrapper.

    Args:
        region_map: 2D numpy array with integer region IDs (0-indexed).
        use_3x3_blocks: Enable the optional 3×3 block constraint.

    Returns:
        List of (row, col) tuples for cat positions, or None if unsolvable.
    """
    solver = MeowdokuSolver(region_map, use_3x3_blocks=use_3x3_blocks)
    return solver.solve()

