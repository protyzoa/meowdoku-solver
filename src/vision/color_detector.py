"""
Color region detector using spatial connected-component analysis.

Instead of global K-means (which ignores cell position), this approach:
  1. Gets the dominant color of each cell in LAB color space
     (perceptually uniform — better for color similarity).
  2. Merges adjacent cells with similar colors into the same region
     using Union-Find.
  3. Auto-calibrates the similarity threshold so the number of regions
     matches the grid size N.

This avoids the classic K-means failure mode where two non-adjacent
same-color areas get merged, or one large region gets split.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from .preprocessor import GridInfo


# ─────────────────────────────────────────────────────────────────
#  Union-Find
# ─────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def num_components(self) -> int:
        return len({self.find(i) for i in range(len(self.parent))})

    def component_map(self) -> List[int]:
        roots = {}
        result = []
        next_id = 0
        for i in range(len(self.parent)):
            r = self.find(i)
            if r not in roots:
                roots[r] = next_id
                next_id += 1
            result.append(roots[r])
        return result


# ─────────────────────────────────────────────────────────────────
#  Per-cell color sampling
# ─────────────────────────────────────────────────────────────────

def _dominant_lab(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Return the dominant color of a cell in CIE-LAB space.
    Ignores very dark pixels (grid lines / shadows) and very light pixels
    (cat icon highlights).
    """
    pixels = cell_bgr.reshape(-1, 3).astype(np.float32)

    # Ignore very dark (borders) and very bright (cat icon / glare)
    brightness = pixels.mean(axis=1)
    mask = (brightness > 40) & (brightness < 230)
    filtered = pixels[mask]
    if len(filtered) < 10:
        filtered = pixels  # fallback: use all

    # Convert to LAB
    bgr_img = filtered.reshape(1, -1, 3).astype(np.uint8)
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    lab = lab_img.reshape(-1, 3).astype(np.float32)
    return lab.mean(axis=0)


# ─────────────────────────────────────────────────────────────────
#  Connected-component region builder
# ─────────────────────────────────────────────────────────────────

def _build_components(
    dom_colors: np.ndarray,   # shape (n, n, 3) in LAB
    n: int,
    threshold: float,
) -> Tuple[np.ndarray, int]:
    """
    Merge adjacent cells whose LAB color distance < threshold.
    Returns (region_map [n,n], num_regions).
    """
    uf = _UnionFind(n * n)

    def idx(r, c): return r * n + c

    for r in range(n):
        for c in range(n):
            color = dom_colors[r, c]
            # Check right and down neighbours only (avoid double-counting)
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    dist = float(np.linalg.norm(color - dom_colors[nr, nc]))
                    if dist < threshold:
                        uf.union(idx(r, c), idx(nr, nc))

    comp_map = uf.component_map()
    region_map = np.array(comp_map, dtype=np.int32).reshape(n, n)
    return region_map, uf.num_components()


def _auto_threshold(
    dom_colors: np.ndarray,
    n: int,
    target_regions: int,
    lo: float = 5.0,
    hi: float = 80.0,
    steps: int = 30,
) -> Tuple[np.ndarray, int, float]:
    """
    Binary-search for the threshold that gives the closest number of
    color regions to `target_regions`.

    Returns (region_map, num_regions, best_threshold).
    """
    best_map = None
    best_count = None
    best_thresh = lo

    for i in range(steps):
        mid = (lo + hi) / 2
        rmap, count = _build_components(dom_colors, n, mid)

        if best_count is None or abs(count - target_regions) < abs(best_count - target_regions):
            best_map = rmap
            best_count = count
            best_thresh = mid

        if count > target_regions:
            lo = mid   # too many regions → raise threshold (merge more)
        elif count < target_regions:
            hi = mid   # too few → lower threshold (merge less)
        else:
            break      # exact match

    return best_map, best_count, best_thresh


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────

def build_region_map(
    grid_info: GridInfo,
    n_colors_hint: Optional[int] = None,
    color_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build an N×N integer region map from the extracted cells using
    spatially-aware connected-component analysis.

    Args:
        grid_info: Output from preprocess_image().
        n_colors_hint: Expected number of color regions. Defaults to N.
        color_threshold: Fixed LAB distance threshold (skip auto-search).

    Returns:
        region_map: np.ndarray of shape (N, N) with integer region IDs.
        region_colors: List of BGR representative colors per region.
    """
    n = grid_info.n
    cells = grid_info.cells
    target = n_colors_hint if n_colors_hint else n

    # ── Step 1: sample dominant LAB color per cell ─────────────────
    dom_colors = np.zeros((n, n, 3), dtype=np.float32)
    for r in range(n):
        for c in range(n):
            dom_colors[r, c] = _dominant_lab(cells[r, c])

    # ── Step 2: find threshold that gives `target` regions ─────────
    if color_threshold is not None:
        region_map, num_regions = _build_components(dom_colors, n, color_threshold)
    else:
        region_map, num_regions, best_thresh = _auto_threshold(
            dom_colors, n, target_regions=target
        )

    # ── Step 3: compute representative BGR colors per region ────────
    num_regions_actual = int(region_map.max()) + 1
    region_colors = []
    for reg_id in range(num_regions_actual):
        mask = (region_map == reg_id)
        idxs = np.argwhere(mask)
        if len(idxs) == 0:
            region_colors.append(np.array([128, 128, 128], dtype=np.uint8))
            continue
        # Average BGR (from original cell crops)
        bgr_vals = []
        for r, c in idxs:
            bgr_vals.append(cells[r, c].reshape(-1, 3).mean(axis=0))
        avg_bgr = np.mean(bgr_vals, axis=0).astype(np.uint8)
        region_colors.append(avg_bgr)

    return region_map, region_colors
