"""
Annotator: Draws the solution on the grid image and prints ASCII board.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import io


def draw_solution_on_image(
    grid_image: np.ndarray,
    solution: List[Tuple[int, int]],
    n: int,
    cell_origins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw cat markers on the grid image at solved positions.

    Args:
        grid_image: BGR image of the grid.
        solution: List of (row, col) tuples.
        n: Grid size.
        cell_origins: (n, n, 2) array of (cy, cx) pixel centers per cell.
                      If None, cell centers are computed from uniform division.

    Returns:
        Annotated BGR image.
    """
    out = grid_image.copy()
    h, w = out.shape[:2]

    for r, c in solution:
        # Use precise cell center from origins if available
        if cell_origins is not None:
            cy, cx = int(cell_origins[r, c, 0]), int(cell_origins[r, c, 1])
        else:
            cell_h = h // n
            cell_w = w // n
            cy = r * cell_h + cell_h // 2
            cx = c * cell_w + cell_w // 2

        # Radius proportional to cell size
        cell_size = min(h, w) // n
        radius = int(cell_size * 0.35)

        # Outer dark ring
        cv2.circle(out, (cx, cy), radius + 2, (30, 30, 30), -1)
        # White fill
        cv2.circle(out, (cx, cy), radius, (255, 255, 255), -1)

        # Cat face emoji via text fallback
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "CAT"
        scale = radius / 38.0
        thickness = max(1, int(scale * 1.5))
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        cv2.putText(out, text, (tx, ty), font, scale, (80, 60, 30), thickness, cv2.LINE_AA)

        # Coordinate label below circle
        coord = f"({r+1},{c+1})"
        lbl_scale = scale * 0.55
        lbl_size = cv2.getTextSize(coord, font, lbl_scale, 1)[0]
        lx = cx - lbl_size[0] // 2
        ly = cy + radius + 14
        cv2.putText(out, coord, (lx, ly), font, lbl_scale, (10, 10, 10), 1, cv2.LINE_AA)

    return out


def draw_solution_pil(
    grid_image: np.ndarray,
    solution: List[Tuple[int, int]],
    n: int,
    cell_origins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    PIL-based renderer that draws cat emoji on the grid.
    Falls back to draw_solution_on_image if emoji rendering fails.
    """
    try:
        rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        h, w = grid_image.shape[:2]
        cell_size = min(h, w) // n

        for r, c in solution:
            if cell_origins is not None:
                cy, cx = int(cell_origins[r, c, 0]), int(cell_origins[r, c, 1])
            else:
                cell_h = h // n
                cell_w = w // n
                cy = r * cell_h + cell_h // 2
                cx = c * cell_w + cell_w // 2

            radius = int(cell_size * 0.38)

            # Semi-transparent white circle
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(255, 255, 255, 210),
                outline=(30, 30, 30, 255),
                width=3,
            )

            # Cat emoji
            font_size = int(radius * 1.1)
            try:
                font = ImageFont.truetype("seguiemj.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("NotoColorEmoji.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

            try:
                draw.text(
                    (cx - font_size // 2, cy - font_size // 2),
                    "\U0001f431",   # 🐱
                    font=font,
                    embedded_color=True,
                )
            except TypeError:
                draw.text(
                    (cx - font_size // 2, cy - font_size // 2),
                    "\U0001f431",
                    font=font,
                    fill=(80, 60, 30, 255),
                )

        # Composite overlay onto original
        combined = Image.alpha_composite(pil_img, overlay).convert("RGB")
        result_bgr = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        return result_bgr

    except Exception:
        return draw_solution_on_image(grid_image, solution, n, cell_origins)


def image_to_bytes(img_bgr: np.ndarray, fmt: str = "PNG") -> bytes:
    """Convert a BGR numpy image to bytes for web response."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def print_ascii_board(
    n: int,
    solution: List[Tuple[int, int]],
    region_map: Optional[np.ndarray] = None,
) -> str:
    """Print a colored ASCII representation of the board."""
    solution_set = set(solution)
    lines = []
    separator = "+" + ("----+" * n)

    lines.append(separator)
    for r in range(n):
        row_str = "|"
        for c in range(n):
            if (r, c) in solution_set:
                cell = " ** "
            else:
                region_id = int(region_map[r, c]) if region_map is not None else -1
                region_char = chr(ord('A') + region_id % 26) if region_id >= 0 else " "
                cell = f"  {region_char} "
            row_str += cell + "|"
        lines.append(row_str)
        lines.append(separator)

    return "\n".join(lines)
