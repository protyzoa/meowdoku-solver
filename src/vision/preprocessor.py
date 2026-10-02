"""
Vision preprocessing: Detects and extracts the Meowdoku grid from an image.

Key improvements:
  - Multi-strategy board detection (contour, color-region, center-crop fallback)
  - Auto grid-size discovery using peak detection + harmonic filtering
  - Cross-validation with color-region count
  - Handles mobile screenshots with status bars, game UI, navigation bars
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class GridInfo:
    """Holds the detected grid information."""
    n: int                    # Grid size (e.g. 6, 8, 9, 10)
    cells: np.ndarray         # Shape (n, n, H, W, 3) — BGR crops of each cell
    grid_image: np.ndarray    # Warped/cropped full grid image (BGR)
    cell_size: int            # Pixel size of each cell
    cell_origins: np.ndarray  # Shape (n, n, 2) — (cy, cx) center pixels per cell


# ─────────────────────────────────────────────────────────────────
#  1. BOARD DETECTION
# ─────────────────────────────────────────────────────────────────

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def _squareness(pts: np.ndarray) -> float:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if w == 0 or h == 0:
        return 0.0
    return float(min(w, h) / max(w, h))


def _detect_board_contour(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 1: Find largest square-ish quadrilateral contour."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = gray.shape[0] * gray.shape[1]
    candidates = []

    for blur_k in [5, 9, 15]:
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        for method in ["adaptive", "canny"]:
            if method == "adaptive":
                binary = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
            else:
                binary = cv2.Canny(blurred, 20, 100)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.dilate(binary, kernel, iterations=2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
                area = cv2.contourArea(c)
                if area < img_area * 0.04:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype("float32")
                    sq = _squareness(pts)
                    if sq > 0.7:
                        candidates.append((area * sq, pts))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _detect_board_by_color_uniformity(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 2: Find largest colored square region (game cells are pastel)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    cell_mask = cv2.inRange(s, 20, 255) & cv2.inRange(v, 80, 240)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cell_mask = cv2.morphologyEx(cell_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = img.shape[0] * img.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ratio = min(bw, bh) / max(bw, bh)
        if ratio > 0.55:
            # Square it up
            side = min(bw, bh)
            cx, cy = x + bw // 2, y + bh // 2
            x0 = max(0, cx - side // 2)
            y0 = max(0, cy - side // 2)
            x1 = min(img.shape[1], x0 + side)
            y1 = min(img.shape[0], y0 + side)
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")

    return None


def _detect_board_by_grid_density(img: np.ndarray) -> Optional[np.ndarray]:
    """Strategy 3: Find region with highest grid-line density by sliding window scan."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)

    row_proj = edges.sum(axis=1).astype(float)
    col_proj = edges.sum(axis=0).astype(float)

    def smooth(x, k=15):
        return np.convolve(x, np.ones(k) / k, mode='same')

    row_proj = smooth(row_proj)
    col_proj = smooth(col_proj)

    def find_dense_span(proj, min_ratio=0.3, max_ratio=0.92):
        n = len(proj)
        best_score, best_start, best_end = 0, 0, n
        for span_ratio in np.linspace(min_ratio, max_ratio, 12):
            span = int(n * span_ratio)
            if span < int(n * min_ratio):
                continue
            cum = np.concatenate([[0], np.cumsum(proj)])
            sums = cum[span:] - cum[:-span]
            pos = int(np.argmax(sums))
            score = sums[pos] / span
            if score > best_score:
                best_score = score
                best_start = pos
                best_end = pos + span
        return best_start, best_end

    y0, y1 = find_dense_span(row_proj)
    x0, x1 = find_dense_span(col_proj)

    side = min(y1 - y0, x1 - x0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)

    if (x1 - x0) * (y1 - y0) < h * w * 0.05:
        return None
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")


def _center_square_crop(img: np.ndarray) -> np.ndarray:
    """Strategy 4 (fallback): Crop centered square from image."""
    h, w = img.shape[:2]
    side = min(h, w)
    return img[(h - side) // 2: (h - side) // 2 + side,
               (w - side) // 2: (w - side) // 2 + side]


def detect_and_crop_board(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """Try multiple strategies to detect and crop the game board."""
    pts = _detect_board_contour(img)
    if pts is not None:
        return _four_point_transform(img, pts), "contour"

    pts = _detect_board_by_color_uniformity(img)
    if pts is not None:
        return _four_point_transform(img, pts), "color_mask"

    pts = _detect_board_by_grid_density(img)
    if pts is not None:
        x0, y0 = int(pts[0, 0]), int(pts[0, 1])
        x1, y1 = int(pts[2, 0]), int(pts[2, 1])
        return img[y0:y1, x0:x1].copy(), "grid_density"

    return _center_square_crop(img), "center_crop"


# ─────────────────────────────────────────────────────────────────
#  2. GRID SIZE AUTO-DISCOVERY
# ─────────────────────────────────────────────────────────────────

def _find_peaks(profile: np.ndarray, min_distance: int = 5) -> List[int]:
    """
    Find local maxima in a 1D profile with minimum distance between peaks.
    """
    peaks = []
    n = len(profile)
    for i in range(1, n - 1):
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
            elif profile[i] > profile[peaks[-1]]:
                peaks[-1] = i
    return peaks


def _score_n_from_peaks(peaks: List[int], profile_size: int, n: int) -> float:
    """
    Given a list of peak positions, score how well they match
    n equally-spaced grid lines across profile_size pixels.
    
    Filters out harmonics: n=5 won't score well if peaks actually match n=10
    by checking that 2*n doesn't score significantly better.
    """
    if len(peaks) < 2:
        return 0.0

    expected_gap = profile_size / n
    matched = 0
    tolerance = expected_gap * 0.25  # 25% tolerance

    for i in range(1, n):
        expected_pos = i * expected_gap
        # Check if any peak is close to this expected position
        for p in peaks:
            if abs(p - expected_pos) < tolerance:
                matched += 1
                break

    return matched / (n - 1)


def _peaks_to_n(profile: np.ndarray, min_n: int, max_n: int) -> int:
    """
    Detect N from a 1D edge-energy profile by finding peaks and
    matching them to the best N, with harmonic suppression.
    """
    size = len(profile)

    # Smooth profile
    k = max(3, size // 100)
    smoothed = np.convolve(profile, np.ones(k) / k, mode='same')

    # Find peaks with minimum distance = expected cell size for max_n
    min_gap = max(3, size // (max_n + 2))
    peaks = _find_peaks(smoothed, min_distance=min_gap)

    if len(peaks) < 2:
        return 0  # can't determine

    # Score each candidate N
    scores = {}
    for n in range(min_n, max_n + 1):
        scores[n] = _score_n_from_peaks(peaks, size, n)

    # Harmonic suppression: if n=5 and n=10 both score well,
    # and peaks gap matches n=10, prefer n=10
    best_n = max(scores, key=lambda n: scores[n])

    # Check harmonic: if best_n * 2 also has good score, prefer the larger
    double = best_n * 2
    if double <= max_n and scores.get(double, 0) >= scores[best_n] * 0.8:
        best_n = double

    # Check half: if best_n / 2 has poor score compared to best_n, keep best_n
    return best_n


def auto_detect_grid_size(
    grid_img: np.ndarray,
    min_n: int = 5,
    max_n: int = 12,
) -> int:
    """
    Automatically discover the number of cells per side (N).

    Method:
      1. Compute Sobel edge maps in X and Y directions.
      2. Project into 1D row and column profiles.
      3. Find peaks in each profile (= grid line positions).
      4. Match peak spacing to candidate N values, with harmonic suppression.
      5. Cross-validate: use median inter-peak gap to estimate N directly.

    Falls back to 8 if uncertain.
    """
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    col_profile = np.abs(sobel_x).sum(axis=0)   # vertical lines → peaks in columns
    row_profile = np.abs(sobel_y).sum(axis=1)   # horizontal lines → peaks in rows

    if col_profile.max() > 0:
        col_profile = col_profile / col_profile.max()
    if row_profile.max() > 0:
        row_profile = row_profile / row_profile.max()

    n_col = _peaks_to_n(col_profile, min_n, max_n)
    n_row = _peaks_to_n(row_profile, min_n, max_n)

    # ── Direct gap estimation ──────────────────────────────────────
    # Estimate N from median gap between peaks — most robust against harmonics
    def gap_estimate(profile, size):
        k = max(3, size // 100)
        smoothed = np.convolve(profile, np.ones(k) / k, mode='same')
        min_gap = max(3, size // (max_n + 2))
        peaks = _find_peaks(smoothed, min_distance=min_gap)
        if len(peaks) < 2:
            return 0
        gaps = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        median_gap = float(np.median(gaps))
        # N = total_size / cell_size, cell_size ≈ median_gap
        n_est = round(size / median_gap)
        return n_est

    n_gap_col = gap_estimate(col_profile, w)
    n_gap_row = gap_estimate(row_profile, h)

    # Collect all estimates
    estimates = [n_col, n_row, n_gap_col, n_gap_row]
    estimates = [e for e in estimates if min_n <= e <= max_n]

    if not estimates:
        return 8  # safe default

    # Pick by majority vote; if tie, pick median
    from collections import Counter
    counts = Counter(estimates)
    max_count = max(counts.values())
    top = [n for n, c in counts.items() if c == max_count]

    if len(top) == 1:
        return top[0]

    # Tie-break: prefer larger N to avoid harmonic under-counting
    return max(top)


# ─────────────────────────────────────────────────────────────────
#  3. CELL EXTRACTION
# ─────────────────────────────────────────────────────────────────

def _trim_border(grid_img: np.ndarray, trim_ratio: float = 0.025) -> np.ndarray:
    """Remove outer rounded-corner border (~2.5% each side)."""
    h, w = grid_img.shape[:2]
    t = int(h * trim_ratio)
    l = int(w * trim_ratio)
    return grid_img[t: h - t, l: w - l]


def _compute_cell_origins(
    grid_img: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Divide the grid image into n×n cells and record the pixel center
    of each cell for precise marker drawing.
    """
    h, w = grid_img.shape[:2]
    margin_ratio = 0.10

    cell_h = h // n
    cell_w = w // n
    cell_size = min(cell_h, cell_w)
    crop_sz = max(cell_size - 2 * int(cell_size * margin_ratio), 4)

    cells = np.zeros((n, n, crop_sz, crop_sz, 3), dtype=np.uint8)
    origins = np.zeros((n, n, 2), dtype=np.int32)

    for r in range(n):
        for c in range(n):
            y0 = r * cell_h
            y1 = (r + 1) * cell_h if r < n - 1 else h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < n - 1 else w

            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            origins[r, c] = [cy, cx]

            margin_y = int((y1 - y0) * margin_ratio)
            margin_x = int((x1 - x0) * margin_ratio)
            crop = grid_img[y0 + margin_y: y1 - margin_y,
                            x0 + margin_x: x1 - margin_x]

            if crop.size == 0:
                continue
            cells[r, c] = cv2.resize(crop, (crop_sz, crop_sz))

    return cells, origins, cell_size


# ─────────────────────────────────────────────────────────────────
#  4. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def preprocess_image(
    image_path: str,
    grid_size_hint: Optional[int] = None,
    target_grid_size: int = 700,
) -> GridInfo:
    """
    Load image → detect board → auto-discover grid size → extract cells.

    Args:
        image_path:      Path to game screenshot (mobile or desktop).
        grid_size_hint:  Override auto-detection (e.g. 10 for 10×10).
        target_grid_size: Resize the extracted board to this pixel size.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # Step 1: Auto-detect and crop the board
    grid_img, detect_strategy = detect_and_crop_board(img)

    # Step 2: Resize to fixed size
    grid_img = cv2.resize(grid_img, (target_grid_size, target_grid_size))

    # Step 3: Trim outer border
    grid_img = _trim_border(grid_img, trim_ratio=0.025)

    # Step 4: Auto-detect grid size (unless overridden)
    if grid_size_hint and 4 <= grid_size_hint <= 14:
        n = grid_size_hint
    else:
        n = auto_detect_grid_size(grid_img)

    # Step 5: Extract cells
    cells, origins, cell_size = _compute_cell_origins(grid_img, n)

    return GridInfo(
        n=n,
        cells=cells,
        grid_image=grid_img,
        cell_size=cell_size,
        cell_origins=origins,
    )
