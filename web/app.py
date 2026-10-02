"""
Flask web app for Meowdoku Solver.
Upload a screenshot -> get the solved puzzle image back.
"""

import os
import sys
import base64
import traceback

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.vision.preprocessor import preprocess_image, detect_and_crop_board, auto_detect_grid_size
from src.vision.color_detector import build_region_map
from src.solver.backtracker import solve_puzzle
from src.renderer.annotator import draw_solution_pil, print_ascii_board, image_to_bytes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    """
    POST /solve
    Form fields:
        - image: file upload
        - grid_size: optional int (e.g. "8")
        - use_3x3: optional "true"/"false"
    Returns JSON:
        - success: bool
        - solution: list of [row, col]
        - ascii_board: string
        - annotated_image: base64 PNG
        - board_preview: base64 PNG (cropped board before annotation)
        - grid_size: int (detected or user-specified)
        - detect_info: dict with detection metadata
        - error: string (on failure)
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    tmp_path = os.path.join(UPLOAD_FOLDER, "upload_tmp.png")
    file.save(tmp_path)

    grid_size_str = request.form.get("grid_size", "").strip()
    grid_size = int(grid_size_str) if grid_size_str.isdigit() else None
    use_3x3 = request.form.get("use_3x3", "false").lower() == "true"

    try:
        # Step 1: Preprocess (board crop + grid size detection)
        grid_info = preprocess_image(tmp_path, grid_size_hint=grid_size)

        # Encode board preview (before annotation)
        board_preview_bytes = image_to_bytes(grid_info.grid_image)
        board_preview_b64 = base64.b64encode(board_preview_bytes).decode("utf-8")

        # Step 2: Color detection
        region_map, region_colors = build_region_map(grid_info)

        # Step 3: Solve
        solution = solve_puzzle(region_map, use_3x3_blocks=use_3x3)

        if solution is None:
            return jsonify({
                "success": False,
                "error": (
                    f"No solution found for detected {grid_info.n}×{grid_info.n} board. "
                    "Try specifying the grid size manually."
                ),
                "grid_size": grid_info.n,
                "board_preview": f"data:image/png;base64,{board_preview_b64}",
            }), 200

        # Step 4: Render annotated image
        annotated = draw_solution_pil(
            grid_info.grid_image, solution, grid_info.n,
            cell_origins=grid_info.cell_origins
        )
        img_bytes = image_to_bytes(annotated)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        ascii_board = print_ascii_board(grid_info.n, solution, region_map)

        return jsonify({
            "success": True,
            "grid_size": grid_info.n,
            "solution": [[r + 1, c + 1] for r, c in solution],
            "ascii_board": ascii_board,
            "annotated_image": f"data:image/png;base64,{img_b64}",
            "board_preview": f"data:image/png;base64,{board_preview_b64}",
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[Meowdoku Solver] Web app running at http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
