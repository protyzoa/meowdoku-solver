# 🐱 Meowdoku Solver

Automatically solve [Meowdoku](https://apps.apple.com/us/app/meowdoku/id6761760135) puzzles from a screenshot.

## How It Works

```
Screenshot → Grid Detection → Color Region Mapping → CSP Solver → Annotated Solution
```

1. **Vision** — OpenCV detects the board boundary (perspective transform), slices it into cells
2. **Color Detection** — K-means clustering groups cells by dominant color → region map
3. **CSP Solver** — Backtracking with forward checking finds valid cat placement
4. **Renderer** — Draws solution on the image and prints an ASCII board

### Constraints Enforced

| Rule | Description |
|------|-------------|
| 1 cat per row | Standard Sudoku-like |
| 1 cat per column | Standard Sudoku-like |
| 1 cat per color region | Meowdoku-specific |
| No adjacency | Cats cannot touch (incl. diagonal) — The Aloof Rule |
| 1 cat per 3×3 block | Optional, enable with `--3x3-blocks` flag |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 🌐 Web App (recommended)

```bash
python web/app.py
```
Then open **http://localhost:5000** in your browser.

- Drag & drop a Meowdoku screenshot
- Optionally specify grid size if auto-detection fails
- Click **Solve Puzzle** → solution appears instantly

### 💻 CLI

```bash
# Basic usage (auto-detect grid size)
python cli.py solve screenshot.png

# With explicit grid size (e.g. 8×8)
python cli.py solve screenshot.png --grid-size 8

# Save solution to custom path
python cli.py solve screenshot.png --output solved.png

# Print ASCII board in terminal
python cli.py solve screenshot.png --show-board

# Show detected regions (for debugging)
python cli.py solve screenshot.png --show-regions

# Enable 3×3 block constraint
python cli.py solve screenshot.png --3x3-blocks
```

---

## Project Structure

```
meowdoku_solver/
├── src/
│   ├── vision/
│   │   ├── preprocessor.py    # Grid detection & cell extraction
│   │   └── color_detector.py  # K-means color clustering
│   ├── solver/
│   │   └── backtracker.py     # CSP backtracking solver
│   └── renderer/
│       └── annotator.py       # Draw solution on image
├── web/
│   ├── app.py                 # Flask server
│   └── templates/index.html   # Web UI
├── tests/
│   └── test_solver.py         # Solver unit tests
├── cli.py                     # CLI entry point
└── requirements.txt
```

---

## Tips for Best Results

- Take a **clean screenshot** of just the puzzle area
- Avoid screenshots with glare, partial views, or UI overlays
- If the grid size is wrong, use `--grid-size N` (CLI) or the grid size field (Web)
- Use `--show-regions` (CLI) to debug color detection

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "N regions but board is N×N" | Color detection found wrong number of regions → specify `--grid-size` correctly |
| No solution found | Board detection may be off → try cropping screenshot to just the grid |
| Wrong region colors | Use `--show-regions` to check detected map visually |

