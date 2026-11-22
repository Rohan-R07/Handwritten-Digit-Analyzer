
import cv2
import numpy as np
import os
import sys

# Add src to path so we can import template_manager
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from template_manager import load_all_templates

CANVAS_SIZE = 28


def _validate_input(image):
    """
    Validate that the input is a 28x28 binary NumPy array with pixel
    values strictly in {0, 255}.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim != 2:
        raise ValueError(
            "Input image must be a single-channel 2D image."
        )

    if image.shape != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(
            "Input image dimensions must be 28x28, got {}x{}.".format(
                image.shape[0], image.shape[1]
            )
        )

    unique_values = set(np.unique(image).tolist())
    if not unique_values <= {0, 255}:
        raise ValueError(
            "Input image pixel values must be strictly 0 or 255."
        )


def _shift_image(image, dx, dy):
    """
    Shift an image by (dx, dy) pixels. Pixels shifted out are lost,
    empty areas are filled with 0 (black).
    """

    shifted = np.zeros_like(image)
    h, w = image.shape

    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)

    dst_y_start = max(0, dy)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, dx)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = (
        image[src_y_start:src_y_end, src_x_start:src_x_end]
    )

    return shifted


def _pixel_similarity(input_image, template):
    """
    Compute pixel-based similarity between two binary images.

    Formula:
        similarity = matching_pixels / total_pixels
    """

    total_pixels = input_image.size
    matching_pixels = int(np.count_nonzero(input_image == template))
    similarity = matching_pixels / total_pixels
    return float(similarity)


def _best_shifted_similarity(input_image, template, max_shift=2):
    """
    Compute the best pixel similarity over small positional shifts of the
    input image relative to the template.
    """

    best = 0.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = _shift_image(input_image, dx, dy)
            score = _pixel_similarity(shifted, template)
            if score > best:
                best = score
    return best


def _shape_similarity(input_image, template):
    """
    Compute shape-based similarity using cv2.matchShapes (Hu moment
    invariants) and distance transform comparison.

    matchShapes returns 0 for identical shapes and higher values for more
    different shapes. We convert this to a 0-1 similarity score using an
    exponential decay: similarity = exp(-match_value / scale).

    Distance transform comparison measures how similar the distance maps
    are between the two images, which is robust to stroke thickness
    variations.

    Returns a value in [0.0, 1.0], where 1.0 means identical shape.
    """

    # --- matchShapes (Hu moment invariants) ---
    # Find contours of the outer boundary.
    contours_input, _ = cv2.findContours(
        input_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_template, _ = cv2.findContours(
        template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_input or not contours_template:
        return 0.0

    # Use the largest contour from each image.
    input_contour = max(contours_input, key=cv2.contourArea)
    template_contour = max(contours_template, key=cv2.contourArea)

    # matchShapes method 1: Hu moments. Returns 0 for identical shapes.
    match_value = cv2.matchShapes(
        input_contour, template_contour, cv2.CONTOURS_MATCH_I1, 0.0
    )

    # Convert match value to similarity [0, 1].
    # matchShapes typically returns values in [0, ~2] for similar shapes,
    # and much higher for dissimilar shapes. Scale factor of 0.5 provides
    # reasonable sensitivity.
    shape_similarity = float(np.exp(-match_value / 0.5))

    # --- Distance transform comparison ---
    # Compute distance to nearest background pixel for each foreground pixel.
    dist_input = cv2.distanceTransform(
        input_image, cv2.DIST_L2, 3
    )
    dist_template = cv2.distanceTransform(
        template, cv2.DIST_L2, 3
    )

    # Normalize distance maps to [0, 1] for comparison.
    max_dist_input = dist_input.max()
    max_dist_template = dist_template.max()

    if max_dist_input > 0 and max_dist_template > 0:
        dist_input_norm = dist_input / max_dist_input
        dist_template_norm = dist_template / max_dist_template

        # Compare normalized distance maps using correlation.
        # Flatten and compute Pearson-like correlation.
        flat_input = dist_input_norm.flatten()
        flat_template = dist_template_norm.flatten()

        mean_input = np.mean(flat_input)
        mean_template = np.mean(flat_template)

        numerator = np.sum(
            (flat_input - mean_input) * (flat_template - mean_template)
        )
        denominator = np.sqrt(
            np.sum((flat_input - mean_input) ** 2) *
            np.sum((flat_template - mean_template) ** 2)
        )

        if denominator > 0:
            dist_correlation = numerator / denominator
            # Correlation is in [-1, 1], convert to [0, 1].
            dist_similarity = (dist_correlation + 1.0) / 2.0
        else:
            dist_similarity = 0.0
    else:
        dist_similarity = 0.0

    # Combine shape and distance similarities.
    combined = 0.6 * shape_similarity + 0.4 * dist_similarity
    return float(combined)


def _best_shifted_shape_similarity(input_image, template, max_shift=2):
    """
    Compute the best shape similarity over small positional shifts.
    """

    best = 0.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = _shift_image(input_image, dx, dy)
            score = _shape_similarity(shifted, template)
            if score > best:
                best = score
    return best


def match_template(input_image):
    """
    Compare an input normalized 28x28 binary digit image against all
    available reference templates and return similarity results.

    Uses a combination of:
      1. Multi-shift pixel similarity (robust to small misalignment)
      2. Shape similarity via cv2.matchShapes (Hu moments, robust to
         stroke thickness and style variations)
      3. Distance transform correlation (robust to stroke thickness)

    Returns pixel similarity as the primary score, but also includes
    shape similarity for use by the hybrid recognizer.
    """

    # Validate the input image.
    _validate_input(input_image)

    # Load all available templates.
    templates = load_all_templates()

    # Handle missing templates gracefully.
    if not templates:
        return {
            "best_digit": None,
            "best_score": None,
            "all_scores": {},
            "ranked_candidates": [],
            "pixel_scores": {},
            "shape_scores": {},
        }

    # Compare the input against every available template.
    pixel_scores = {}
    shape_scores = {}
    combined_scores = {}

    for digit in sorted(templates.keys()):
        template = templates[digit]

        pixel_score = _best_shifted_similarity(input_image, template)
        shape_score = _best_shifted_shape_similarity(input_image, template)

        # Combine pixel and shape scores.
        # Weight pixel similarity at 40%, shape at 60% - shape is more
        # robust to stroke style variations than raw pixel matching.
        combined = 0.4 * pixel_score + 0.6 * shape_score

        # Anti-digit-1 penalty: a simple vertical line template matches
        # almost everything pixel-wise (high pixel score), but its shape
        # similarity is typically low for non-vertical digits. When pixel
        # score is high but shape score is low for digit 1, reduce the
        # combined score aggressively.
        if digit == 1 and pixel_score > 0.55 and shape_score < 0.50:
            combined = combined * (0.4 + 0.6 * shape_score)

        pixel_scores[digit] = pixel_score
        shape_scores[digit] = shape_score
        combined_scores[digit] = combined

    # Rank candidates by descending combined score.
    ranked_candidates = sorted(
        combined_scores.items(), key=lambda item: item[1], reverse=True
    )

    # Best match is the first entry in the ranked list.
    best_digit, best_score = ranked_candidates[0]

    return {
        "best_digit": int(best_digit),
        "best_score": float(best_score),
        "all_scores": combined_scores,
        "ranked_candidates": ranked_candidates,
        "pixel_scores": pixel_scores,
        "shape_scores": shape_scores,
    }
