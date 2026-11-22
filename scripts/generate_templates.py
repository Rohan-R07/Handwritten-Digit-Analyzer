
import cv2
import numpy as np
import os
import sys

# Add src to path so we can import template_manager
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from template_manager import (
    load_all_templates,
    get_missing_templates,
    validate_template_file,
    TEMPLATE_DIR,
    CANVAS_SIZE,
)

CANVAS_SIZE = 28
STROKE_THICKNESS = 3
CENTER = CANVAS_SIZE // 2  # 14


def _blank_canvas():
    """Create a black 28x28 single-channel canvas."""
    return np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)


def _draw_0(canvas):
    """Draw digit 0: centered oval outline."""
    cv2.ellipse(
        canvas,
        (CENTER, CENTER),
        (10, 13),
        0,
        0,
        360,
        255,
        STROKE_THICKNESS,
    )
    return canvas


def _draw_1(canvas):
    """Draw digit 1: vertical line with small base serif."""
    cv2.line(canvas, (CENTER, 5), (CENTER, CANVAS_SIZE - 5), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER - 4, CANVAS_SIZE - 5), (CENTER + 4, CANVAS_SIZE - 5), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER, 5), (CENTER - 2, 9), 255, STROKE_THICKNESS)
    return canvas


def _draw_2(canvas):
    """Draw digit 2: top curve, diagonal, horizontal base."""
    # Top curve
    cv2.ellipse(canvas, (CENTER, 8), (9, 6), 0, 0, 180, 255, STROKE_THICKNESS)
    # Diagonal from top-right to bottom-left
    cv2.line(canvas, (CENTER + 8, 14), (CENTER - 8, CANVAS_SIZE - 6), 255, STROKE_THICKNESS)
    # Horizontal base
    cv2.line(canvas, (CENTER - 8, CANVAS_SIZE - 5), (CENTER + 8, CANVAS_SIZE - 5), 255, STROKE_THICKNESS)
    return canvas


def _draw_3(canvas):
    """Draw digit 3: two curves on the right side."""
    # Top curve
    cv2.ellipse(canvas, (CENTER + 4, 10), (8, 6), 0, 0, 180, 255, STROKE_THICKNESS)
    # Bottom curve
    cv2.ellipse(canvas, (CENTER + 4, 18), (8, 6), 0, 0, 180, 255, STROKE_THICKNESS)
    # Connecting spine on right
    cv2.line(canvas, (CENTER + 11, 10), (CENTER + 11, 18), 255, STROKE_THICKNESS)
    return canvas


def _draw_4(canvas):
    """Draw digit 4: vertical right, horizontal middle, short vertical top-left."""
    cv2.line(canvas, (CENTER + 4, CANVAS_SIZE - 5), (CENTER + 4, 5), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER - 4, 12), (CENTER + 8, 12), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER - 4, 5), (CENTER - 4, 12), 255, STROKE_THICKNESS)
    return canvas


def _draw_5(canvas):
    """Draw digit 5: horizontal top, vertical down, bottom curve."""
    cv2.line(canvas, (CENTER - 8, 6), (CENTER + 8, 6), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER - 8, 6), (CENTER - 8, 13), 255, STROKE_THICKNESS)
    cv2.ellipse(canvas, (CENTER + 2, 14), (8, 6), 180, 0, 180, 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER + 8, 20), (CENTER - 2, CANVAS_SIZE - 5), 255, STROKE_THICKNESS)
    return canvas


def _draw_6(canvas):
    """Draw digit 6: loop at bottom with curve from top."""
    cv2.ellipse(canvas, (CENTER, CANVAS_SIZE - 8), (9, 8), 0, 0, 360, 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER, CANVAS_SIZE - 16), (CENTER, 6), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER, 6), (CENTER - 6, 10), 255, STROKE_THICKNESS)
    return canvas


def _draw_7(canvas):
    """Draw digit 7: horizontal top with diagonal down-left."""
    cv2.line(canvas, (CENTER - 8, 6), (CENTER + 8, 6), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER + 8, 6), (CENTER - 6, CANVAS_SIZE - 5), 255, STROKE_THICKNESS)
    return canvas


def _draw_8(canvas):
    """Draw digit 8: two stacked loops."""
    cv2.ellipse(canvas, (CENTER, 10), (9, 7), 0, 0, 360, 255, STROKE_THICKNESS)
    cv2.ellipse(canvas, (CENTER, 18), (9, 7), 0, 0, 360, 255, STROKE_THICKNESS)
    return canvas


def _draw_9(canvas):
    """Draw digit 9: loop at top with curve coming down from bottom."""
    cv2.ellipse(canvas, (CENTER, 10), (9, 8), 0, 0, 360, 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER, 18), (CENTER, CANVAS_SIZE - 5), 255, STROKE_THICKNESS)
    cv2.line(canvas, (CENTER, CANVAS_SIZE - 5), (CENTER + 6, CANVAS_SIZE - 9), 255, STROKE_THICKNESS)
    return canvas


DRAWERS = {
    0: _draw_0,
    1: _draw_1,
    2: _draw_2,
    3: _draw_3,
    4: _draw_4,
    5: _draw_5,
    6: _draw_6,
    7: _draw_7,
    8: _draw_8,
    9: _draw_9,
}


def generate_templates(force=False):
    """
    Generate reference digit templates for digits 0-9.

    Parameters
    ----------
    force : bool
        If True, overwrite existing template files. If False, skip files
        that already exist.

    Returns
    -------
    dict
        Mapping of digit -> {"status": str, "path": str or None, "message": str}
    """

    results = {}

    for digit in range(10):
        path = os.path.join(TEMPLATE_DIR, "{}.png".format(digit))

        if os.path.isfile(path) and not force:
            results[digit] = {
                "status": "skipped",
                "path": path,
                "message": "File already exists. Use force=True to overwrite.",
            }
            continue

        # Draw the digit
        canvas = _blank_canvas()
        drawer = DRAWERS[digit]
        canvas = drawer(canvas)

        # Verify it's strictly binary before saving
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)

        # Save
        success = cv2.imwrite(path, canvas)

        if not success:
            results[digit] = {
                "status": "failed",
                "path": path,
                "message": "cv2.imwrite returned False.",
            }
            continue

        # Validate using template_manager
        validation = validate_template_file(path)

        if validation["valid"]:
            results[digit] = {
                "status": "success",
                "path": path,
                "message": "Template generated and validated successfully.",
            }
        else:
            results[digit] = {
                "status": "failed",
                "path": path,
                "message": "Validation failed: {}".format(validation["reason"]),
            }

    return results


def main():
    """CLI entry point to generate all digit templates."""
    print("=" * 70)
    print("DIGIT TEMPLATE GENERATOR")
    print("=" * 70)
    print()

    # Check for existing templates
    existing = get_missing_templates()
    if len(existing) < 10:
        print("Existing templates detected for digits: {}".format(
            [d for d in range(10) if d not in existing]
        ))
        print("Use --force to overwrite existing templates.")
        print()

    results = generate_templates(force=False)

    print("{:<6} {:<12} {:<30} {}".format("Digit", "Status", "File", "Message"))
    print("-" * 70)

    all_success = True
    for digit in range(10):
        r = results[digit]
        status = r["status"]
        if status != "success":
            all_success = False
        print("{:<6} {:<12} {:<30} {}".format(
            digit, status, os.path.basename(r["path"] or "N/A"), r["message"]
        ))

    print()
    print("=" * 70)

    # Final verification using load_all_templates
    print("Final verification with load_all_templates():")
    templates = load_all_templates()
    loaded_digits = sorted(templates.keys())
    print("Loaded digit keys: {}".format(loaded_digits))

    missing = get_missing_templates()
    print("Missing digits: {}".format(missing))

    if missing:
        print("WARNING: {} template(s) missing or invalid.".format(len(missing)))
        all_success = False
    else:
        print("All 10 templates loaded and validated successfully.")

    # Verify each template passes validation
    for digit in range(10):
        path = os.path.join(TEMPLATE_DIR, "{}.png".format(digit))
        if os.path.isfile(path):
            v = validate_template_file(path)
            if not v["valid"]:
                print("WARNING: {} failed validation: {}".format(path, v["reason"]))
                all_success = False

    print()
    if all_success:
        print("RESULT: All templates generated and validated successfully.")
    else:
        print("RESULT: Some templates failed. Review messages above.")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
