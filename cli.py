"""
CLI entry point for Meowdoku Solver.

Usage:
    python cli.py solve path/to/screenshot.png
    python cli.py solve path/to/screenshot.png --grid-size 8 --output solved.png
    python cli.py solve path/to/screenshot.png --show-board
"""

import sys
import os
import click
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.vision.preprocessor import preprocess_image
from src.vision.color_detector import build_region_map
from src.solver.backtracker import solve_puzzle
from src.renderer.annotator import draw_solution_pil, print_ascii_board, image_to_bytes


@click.group()
def cli():
    """🐱 Meowdoku Solver — Upload a screenshot, get the solution!"""
    pass


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--grid-size", "-n", type=int, default=None,
              help="Override grid size detection (e.g. 6 for 6×6)")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save annotated solution image to this path")
@click.option("--show-board", is_flag=True,
              help="Print ASCII board to terminal")
@click.option("--3x3-blocks", "use_3x3", is_flag=True,
              help="Enable optional 3×3 block constraint")
@click.option("--show-regions", is_flag=True,
              help="Show the detected region map before solving")
def solve(image_path, grid_size, output, show_board, use_3x3, show_regions):
    """Solve a Meowdoku puzzle from a screenshot image."""

    click.echo(f"📷  Loading image: {image_path}")

    # ── Step 1: Vision preprocessing ────────────────────────────────────────
    try:
        grid_info = preprocess_image(image_path, grid_size_hint=grid_size)
    except FileNotFoundError as e:
        click.echo(f"❌  {e}", err=True)
        sys.exit(1)

    click.echo(f"🔍  Detected grid size: {grid_info.n}×{grid_info.n}")

    # ── Step 2: Color region detection ──────────────────────────────────────
    click.echo("🎨  Detecting color regions...")
    try:
        region_map, region_colors = build_region_map(grid_info)
    except Exception as e:
        click.echo(f"❌  Color detection failed: {e}", err=True)
        sys.exit(1)

    if show_regions:
        click.echo("\n📊  Region map (letters = region IDs):")
        for r in range(grid_info.n):
            row = " ".join(chr(ord('A') + int(region_map[r, c])) for c in range(grid_info.n))
            click.echo(f"  {row}")
        click.echo()

    # ── Step 3: Solve ────────────────────────────────────────────────────────
    click.echo("🧩  Solving puzzle...")
    try:
        solution = solve_puzzle(region_map, use_3x3_blocks=use_3x3)
    except ValueError as e:
        click.echo(f"❌  Solver error: {e}", err=True)
        click.echo(
            "\n💡  Tip: The number of detected color regions must equal the grid size.\n"
            "    Try specifying --grid-size manually if auto-detection is off.",
            err=True
        )
        sys.exit(1)

    if solution is None:
        click.echo("❌  No solution found. The puzzle may be unsolvable with current region detection.")
        click.echo("    Try --show-regions to inspect detected regions, or --grid-size N to override.")
        sys.exit(1)

    click.echo(f"\n✅  Solution found! Cat positions (row, col):")
    for r, c in solution:
        click.echo(f"    Row {r+1}, Column {c+1}")

    # ── Step 4: Render ───────────────────────────────────────────────────────
    if show_board:
        click.echo("\n🏠  Board:\n")
        ascii_board = print_ascii_board(grid_info.n, solution, region_map)
        click.echo(ascii_board)

    annotated = draw_solution_pil(grid_info.grid_image, solution, grid_info.n,
                                   cell_origins=grid_info.cell_origins)

    if output:
        cv2.imwrite(output, annotated)
        click.echo(f"\n💾  Saved annotated image to: {output}")
    else:
        # Auto-save next to source
        base, ext = os.path.splitext(image_path)
        auto_out = base + "_solved.png"
        cv2.imwrite(auto_out, annotated)
        click.echo(f"\n💾  Saved annotated image to: {auto_out}")

    # Show with OpenCV if running interactively
    try:
        cv2.imshow("Meowdoku Solution", annotated)
        click.echo("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass  # Headless environment


if __name__ == "__main__":
    cli()

