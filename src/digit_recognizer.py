
import cv2
import numpy as np
import os
import sys

# Add src to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from template_matcher import match_template
from feature_extraction import extract_all_features, extract_normalized_features
from rule_classifier import classify_features

# Minimum template similarity to consider the input a digit at all.
MIN_TEMPLATE_SIMILARITY = 0.40

# Template score threshold for "strong" evidence.
STRONG_TEMPLATE_THRESHOLD = 0.70

# If the top two template scores are within this margin, feature scores
# are allowed to break the tie. Otherwise template evidence dominates.
TEMPLATE_TIE_MARGIN = 0.08


def recognize_digit(cropped_binary, crop_info=None, min_hole_area=20):
    """
    Hybrid digit recognition using template matching + geometric features.

    Template matching is the primary signal. Geometric features are used
    as a tiebreaker when template scores are close, and as supporting
    context. Strong structural evidence (e.g. two vertically-stacked holes
    for digit 8, upper curve + diagonal + base for digit 2, two curves for
    digit 3) can boost the corresponding digit's score.

    Decision tiers:
      - "ok" (Recognized): strong combined evidence with clear separation.
      - "uncertain": moderate evidence, best candidate reported with warning.
      - "invalid_input": weak evidence or clearly non-digit input.

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Normalized 28x28 binary digit image (0 = background, 255 = foreground).
    crop_info : dict or None
        Optional dictionary from crop_digit(). Passed to feature extraction
        for bounding box reuse.
    min_hole_area : int
        Minimum area for hole detection. Default is 20.

    Returns
    -------
    dict
        {
            "predicted_digit": int or None,
            "similarity_score": float or None,
            "alternative_candidates": [(digit, score), ...],
            "template_scores": {digit: score},
            "shape_scores": {digit: score},
            "feature_scores": {digit: score} or None,
            "explanation": str,
            "status": "ok" | "uncertain" | "invalid_input" | "no_templates" | "error",
            "error_message": str or None
        }
    """

    result = {
        "predicted_digit": None,
        "similarity_score": None,
        "alternative_candidates": [],
        "template_scores": {},
        "shape_scores": {},
        "feature_scores": None,
        "explanation": "",
        "status": "error",
        "error_message": None,
    }

    # --- Validate input ---
    if not isinstance(cropped_binary, np.ndarray):
        result["error_message"] = "cropped_binary must be a NumPy array."
        return result

    if cropped_binary.size == 0:
        result["error_message"] = "cropped_binary is empty."
        return result

    if cropped_binary.ndim != 2:
        result["error_message"] = "cropped_binary must be a 2D single-channel image."
        return result

    if cropped_binary.shape != (28, 28):
        result["error_message"] = (
            "cropped_binary must be 28x28, got {}x{}.".format(
                cropped_binary.shape[0], cropped_binary.shape[1]
            )
        )
        return result

    unique_values = set(np.unique(cropped_binary).tolist())
    if not unique_values <= {0, 255}:
        result["error_message"] = "cropped_binary pixel values must be strictly 0 or 255."
        return result

    # Reject blank or nearly-blank images (no meaningful foreground).
    foreground_fraction = float(np.count_nonzero(cropped_binary == 255)) / float(cropped_binary.size)
    if foreground_fraction < 0.01:
        result["status"] = "invalid_input"
        result["error_message"] = (
            "Image contains no meaningful foreground content (less than 1% "
            "foreground pixels). Cannot recognize a digit."
        )
        result["explanation"] = (
            "Rejected: blank or nearly-blank image. No digit visible."
        )
        return result

    # --- Step 1: Template matching (primary signal) ---
    # Make a copy to protect against cv2.findContours modifying the input
    # image in-place.
    cropped_binary_for_match = cropped_binary.copy()
    template_result = match_template(cropped_binary_for_match)

    # Handle missing templates.
    if not template_result or template_result.get("best_digit") is None:
        result["status"] = "no_templates"
        result["error_message"] = (
            "No reference templates available. Cannot perform recognition."
        )
        result["explanation"] = (
            "Recognition skipped: no reference digit templates found in "
            "the templates/ directory."
        )
        return result

    template_scores = template_result["all_scores"]
    shape_scores = template_result["shape_scores"]
    best_template_digit = template_result["best_digit"]
    best_template_score = template_result["best_score"]
    ranked_candidates = template_result["ranked_candidates"]

    result["template_scores"] = {d: round(float(s), 4) for d, s in template_scores.items()}
    result["shape_scores"] = {d: round(float(s), 4) for d, s in template_result["shape_scores"].items()}

    # Reject if the best template similarity is below threshold.
    # This prevents arbitrary images from being matched to a digit just
    # because feature scores happen to be high.
    if best_template_score < MIN_TEMPLATE_SIMILARITY:
        result["status"] = "invalid_input"
        result["error_message"] = (
            "Best template similarity {:.2f} is below minimum threshold "
            "{:.2f}. Input does not resemble any reference digit."
        ).format(best_template_score, MIN_TEMPLATE_SIMILARITY)
        result["explanation"] = (
            "Rejected: input does not sufficiently resemble any reference "
            "digit template (best similarity {:.2f})."
        ).format(best_template_score)
        return result

    # --- Step 2: Feature extraction (supporting/tiebreaker signal) ---
    feature_scores = None
    feature_explanation = ""
    classification = None

    # Make copies to protect against cv2.findContours modifying images
    # in-place during connected component analysis.
    cropped_binary_for_all = cropped_binary.copy()
    cropped_binary_for_normalized = cropped_binary.copy()

    # Compute hole features directly using connected components (does not
    # modify the input image, unlike findContours).
    computed_min_hole_area = max(1, min_hole_area // 100)
    cropped_binary_for_all = cropped_binary.copy()
    cropped_binary_for_normalized = cropped_binary.copy()

    # Compute hole features directly using connected components analysis
    # (does not modify the input image).
    num_labels, hole_labels, hole_stats, hole_centroids = cv2.connectedComponentsWithStats(
        cropped_binary.copy(), connectivity=8
    )
    hole_positions = []
    hole_areas = []
    hole_count = 0
    for i in range(1, num_labels):
        area = float(hole_stats[i, cv2.CC_STAT_AREA])
        if area >= computed_min_hole_area:
            cx = float(hole_centroids[i][0]) / cropped_binary.shape[1]
            cy = float(hole_centroids[i][1]) / cropped_binary.shape[1]
            hole_positions.append((cx, cy))
            hole_areas.append(float(area))
            hole_count += 1

    try:
        all_features = extract_all_features(
            cropped_binary_for_all, crop_info=crop_info, min_hole_area=computed_min_hole_area
        )
        normalized = extract_normalized_features(
            cropped_binary_for_normalized, crop_info=crop_info, min_hole_area=computed_min_hole_area
        )

        classification = classify_features(normalized)

        if classification["status"] == "ok":
            feature_scores = classification["scores"]
            result["feature_scores"] = {
                d: round(float(s), 4) for d, s in feature_scores.items()
            }

            # Build feature explanation from top candidates
            top_feature = classification["ranked_candidates"][0]
            feature_explanation = (
                "Geometric features suggest digit {} "
                "(score: {:.2f}).".format(
                    top_feature[0], top_feature[1]
                )
            )
        else:
            feature_explanation = (
                "Geometric feature classification skipped: {}."
            ).format(classification.get("error_message", "unknown"))

    except Exception as feature_error:
        feature_explanation = (
            "Geometric feature extraction failed: {}."
        ).format(feature_error)

    # --- Step 3: Combine template and feature signals ---
    # Template is the primary signal (pixel + shape combined). Features
    # provide a tiebreaker when template scores are close. Strong structural
    # evidence can boost specific digits.

    combined_scores = {}

    template_top1 = ranked_candidates[0][1]
    template_top2 = ranked_candidates[1][1] if len(ranked_candidates) > 1 else 0.0
    template_is_close = (template_top1 - template_top2) <= TEMPLATE_TIE_MARGIN

    # --- Structural evidence detection ---
    # Detect strong structural evidence for specific digits based on hole
    # positions, projection profiles, and aspect ratio.
    strong_structural = {}  # digit -> bool
    strong_structural_confidence = {}  # digit -> float confidence 0-1
    structural_reasons = {}  # digit -> reason string

    if normalized and feature_scores:
        aspect_ratio = normalized.get("aspect_ratio", 1.0)
        foreground_density = normalized.get("foreground_density", 0.0)
        h_projection = normalized.get("normalized_horizontal_projection", None)
        v_projection = normalized.get("normalized_vertical_projection", None)

        # Pre-compute projection metrics (always available for anti-digit-1 check).
        h_top_heavy = _projection_top_heavy_ratio(h_projection)
        v_left_heavy = _projection_left_heavy_ratio(v_projection)
        v_peak_pos = _projection_peak_position(v_projection, "vertical")

        # Detect rising pattern at top of digit (curve going up from top edge).
        # Digit 2 has a curve at top (rising projection), while digit 7 has a
        # flat top bar. Computed within the digit's non-empty vertical range.
        rising_at_top = False
        if h_projection is not None and len(h_projection) > 5:
            h_arr = np.asarray(h_projection, dtype=np.float64)
            non_zero = np.nonzero(h_arr)[0]
            if len(non_zero) > 5:
                d_top = non_zero[0]
                d_mid = (d_top + len(h_arr) - 1) // 2
                top_region = h_arr[d_top:d_mid]
                if len(top_region) > 3:
                    first_quarter = np.mean(top_region[:len(top_region) // 4])
                    last_quarter = np.mean(top_region[len(top_region) * 3 // 4:])
                    if last_quarter > first_quarter * 1.5 and last_quarter > 0.15:
                        rising_at_top = True

        # --- Digit 8: two vertically-stacked, balanced holes ---
        strong_structural[8] = False
        if hole_count >= 2 and len(hole_positions) >= 2:
            sorted_areas = sorted(hole_areas, reverse=True)
            y_positions = sorted([pos[1] for pos in hole_positions])

            two_significant = False
            if len(hole_areas) == 2:
                two_significant = True
            elif len(hole_areas) >= 3:
                if sorted_areas[1] > 0 and sorted_areas[2] < 0.5 * sorted_areas[1]:
                    if y_positions[0] < 0.45 and y_positions[-1] > 0.55:
                        two_significant = True

            vertically_arranged = (
                two_significant
                and y_positions[0] < 0.45
                and y_positions[-1] > 0.55
            )

            areas_balanced = False
            if len(hole_areas) >= 2:
                a1, a2 = sorted_areas[0], sorted_areas[1]
                if a1 > 0 and a2 > 0:
                    ratio = min(a1, a2) / max(a1, a2)
                    areas_balanced = ratio > 0.4

            strong_structural[8] = two_significant and vertically_arranged and areas_balanced
            if strong_structural[8]:
                structural_reasons[8] = "two vertically-stacked, balanced enclosed holes"

        # --- Digit 2: curve top, diagonal transition, horizontal base ---
        # Characteristics: no holes, top-heavy horizontal projection
        # (curve at top), moderate density, near-square aspect ratio.
        strong_structural[2] = False
        if hole_count == 0 and (h_top_heavy > 1.05 or rising_at_top):
            strong_structural[2] = True
            structural_reasons[2] = "curve at top, diagonal transition, horizontal base"

        # --- Digit 3: two curved lobes on right side ---
        # Characteristics: no holes, right-heavy vertical projection
        # (curves on right), moderate density, near-square aspect ratio.
        strong_structural[3] = False
        if hole_count == 0 and v_left_heavy < 0.90:
            strong_structural[3] = True
            structural_reasons[3] = "two curved lobes on right side"

        # --- Digit 0: single centered enclosed hole ---
        strong_structural[0] = False
        strong_structural_confidence[0] = 0.0
        if hole_count == 1:
            # Check if hole is centered.
            if len(hole_positions) == 1:
                hx, hy = hole_positions[0]
                ha = hole_areas[0] if hole_areas else 0.0
                if abs(hx - 0.5) < 0.2 and abs(hy - 0.5) < 0.2:
                    # Hole must be large enough to be genuine (not just a gap
                    # between strokes).
                    if ha > 0.15:
                        strong_structural[0] = True
                        # Confidence based on hole size and centering
                        center_dist = abs(hx - 0.5) + abs(hy - 0.5)
                        strong_structural_confidence[0] = max(0.0, ha * (1.0 - center_dist))
                        structural_reasons[0] = "single centered enclosed hole"

        # --- Digit 6: single hole at bottom ---
        strong_structural[6] = False
        if hole_count == 1:
            if len(hole_positions) == 1:
                hx, hy = hole_positions[0]
                ha = hole_areas[0] if hole_areas else 0.0
                # Hole must be in lower half AND large enough to be genuine
                # (not just a gap between strokes).
                if hy > 0.55 and ha > 0.15:
                    strong_structural[6] = True
                    structural_reasons[6] = "single enclosed hole at bottom"

        # --- Digit 9: single hole at top ---
        strong_structural[9] = False
        if hole_count == 1:
            if len(hole_positions) == 1:
                hx, hy = hole_positions[0]
                ha = hole_areas[0] if hole_areas else 0.0
                # Hole must be in upper half AND large enough to be genuine.
                if hy < 0.45 and ha > 0.15:
                    strong_structural[9] = True
                    structural_reasons[9] = "single enclosed hole at top"

        # --- Compute combined scores ---
        combined_scores = {}

        for digit in range(10):
            template_score = template_scores.get(digit, 0.0)
            feature_score = feature_scores.get(digit, 0.0) if feature_scores else 0.0

            # Determine if structural evidence should boost this digit.
            has_strong_structure = strong_structural.get(digit, False)

            if has_strong_structure:
                # Strong structural evidence: give features more weight.
                # But only if template score is at least reasonable (> 0.30)
                # to avoid classifying random shapes as a specific digit.
                if template_score > 0.30:
                    combined = 0.2 * template_score + 0.75 * feature_score
                else:
                    # Template is too weak even with structural evidence.
                    combined = 0.50 * template_score + 0.50 * feature_score
            elif template_is_close:
                # Template scores are close: features provide strong signal.
                combined = 0.3 * template_score + 0.7 * feature_score
            else:
                # Template evidence is unambiguous: features still matter.
                combined = 0.5 * template_score + 0.5 * feature_score

            # --- Anti-digit-1 penalty ---
            # Digit 1 (simple vertical line) matches almost everything via pixel
            # similarity. When there's clear evidence of curves, diagonals, or
            # non-vertical strokes, penalize digit 1's score to prevent false
            # positives. Also penalize when shape similarity for digit 1 is low
            # (shape doesn't match a simple vertical line).
            if digit == 1 and feature_scores:
                anti_digit1_penalty = 0.0
                # Curve at top (top-heavy horizontal projection)
                if h_top_heavy > 1.05:
                    anti_digit1_penalty += 0.25
                # Horizontal base at bottom (bottom-heavy horizontal projection)
                if h_top_heavy < 0.90:
                    anti_digit1_penalty += 0.25
                # Not tall enough for digit 1 (aspect ratio > 0.5 means width is
                # more than half of height, too wide for a simple vertical stroke)
                if aspect_ratio > 0.5:
                    anti_digit1_penalty += 0.20
                # Too dense for digit 1 (should be a thin stroke)
                if foreground_density > 0.18:
                    anti_digit1_penalty += 0.20
                # Vertical projection not centered (digit 1 should be centered)
                if abs(v_peak_pos - 0.5) > 0.10:
                    anti_digit1_penalty += 0.15
                # Shape similarity for digit 1 is low (doesn't match vertical line)
                shape_score_1 = shape_scores.get(1, 1.0) if shape_scores else 1.0
                if shape_score_1 < 0.40:
                    anti_digit1_penalty += 0.20
                combined = max(0.0, combined - anti_digit1_penalty)

            # --- Shape-based boost for digits 2 and 3 ---
            # Shape similarity (cv2.matchShapes + distance transform) is more
            # robust to stroke style variations than pixel matching. When shape
            # similarity for digit 2 or 3 is high, boost their score even if
            # pixel matching favors digit 1.
            shape_score = shape_scores.get(digit, 0.0) if shape_scores else 0.0
            shape_boost = 0.0
            if digit == 2 and shape_score > 0.55:
                shape_boost = 0.10 * shape_score
            elif digit == 3 and shape_score > 0.55:
                shape_boost = 0.10 * shape_score
            elif digit == 0 and strong_structural.get(0, False) and shape_score > 0.45:
                shape_boost = 0.10 * shape_score

            combined = combined + shape_boost

            combined_scores[digit] = combined

        # Rank combined candidates
        combined_ranked = sorted(
            combined_scores.items(), key=lambda item: item[1], reverse=True
        )

        top_digit, top_score = combined_ranked[0]
        second_digit, second_score = combined_ranked[1]

        # Check separation between top two combined candidates.
        separation = top_score - second_score

        # --- Step 4: Determine decision tier ---
        if top_score >= STRONG_TEMPLATE_THRESHOLD and separation >= 0.05:
            # Strong evidence: clear template match, well-separated from runner-up.
            result["status"] = "ok"
            result["explanation"] = (
                "Recognized with strong template match. "
                "Template similarity for digit {}: {:.2f}. Combined score: {:.2f}."
            ).format(top_digit, top_score, round(float(top_score), 4))
        elif top_score >= MIN_TEMPLATE_SIMILARITY:
            # Moderate evidence: template match is acceptable but not decisive.
            result["status"] = "uncertain"
            result["explanation"] = (
                "Uncertain prediction: template match is moderate "
                "(similarity {:.2f}). Top candidate is digit {} "
                "(combined score: {:.2f}), but separation from runner-up "
                "is small ({:.2f})."
            ).format(
                top_score,
                int(top_digit),
                round(float(top_score), 4),
                round(float(separation), 4),
            )
        else:
            # Very weak evidence: reject.
            result["status"] = "invalid_input"
            result["error_message"] = (
                "Best similarity {:.2f} is too low for reliable "
                "recognition."
            ).format(top_score)
            result["explanation"] = (
                "Rejected: input does not sufficiently resemble any reference "
                "digit template (best similarity {:.2f})."
            ).format(top_score)

        # --- Final result ---
        result["predicted_digit"] = int(top_digit)
        result["similarity_score"] = round(float(top_score), 4)
        result["alternative_candidates"] = [
            (int(d), round(float(s), 4))
            for d, s in combined_ranked[1:4]
        ]
        result["ranked_candidates"] = [
            (int(d), round(float(s), 4)) for d, s in combined_ranked
        ]

        # Add feature agreement/disagreement note to explanation
        if feature_scores and result["status"] in ("ok", "uncertain"):
            feature_top = classification["ranked_candidates"][0]
            if feature_top[0] == int(top_digit):
                result["explanation"] += " Geometric features agree."
            else:
                result["explanation"] += " Geometric features suggest digit {} instead.".format(
                    feature_top[0]
                )

        # Add structural evidence note when it influenced the decision.
        if strong_structural.get(int(top_digit), False) and result["status"] in ("ok", "uncertain"):
            result["explanation"] += (
                " Structural evidence for digit {}: {}.".format(
                    top_digit, structural_reasons.get(int(top_digit), "")
                )
            )

    return result


def _projection_top_heavy_ratio(projection):
    """
    Ratio of projection mass in the top half vs bottom half.
    Returns >1 if top-heavy, <1 if bottom-heavy, ~1 if balanced.
    """
    if projection is None or len(projection) == 0:
        return 1.0
    projection = np.asarray(projection, dtype=np.float64)
    total = np.sum(projection)
    if total == 0:
        return 1.0
    mid = len(projection) // 2
    top_mass = np.sum(projection[:mid])
    bottom_mass = np.sum(projection[mid:])
    if bottom_mass == 0:
        return float('inf')
    return float(top_mass) / float(bottom_mass)


def _projection_left_heavy_ratio(projection):
    """
    Ratio of projection mass in the left half vs right half.
    Returns >1 if left-heavy, <1 if right-heavy, ~1 if balanced.
    """
    if projection is None or len(projection) == 0:
        return 1.0
    projection = np.asarray(projection, dtype=np.float64)
    total = np.sum(projection)
    if total == 0:
        return 1.0
    mid = len(projection) // 2
    left_mass = np.sum(projection[:mid])
    right_mass = np.sum(projection[mid:])
    if right_mass == 0:
        return float('inf')
    return float(left_mass) / float(right_mass)


def _projection_peak_position(projection, axis_name):
    """
    Return the normalized position (0.0 to 1.0) of the peak in a projection
    profile. Returns 0.5 if projection is empty or all zeros.
    """
    if projection is None or len(projection) == 0:
        return 0.5
    projection = np.asarray(projection, dtype=np.float64)
    if np.max(projection) == 0:
        return 0.5
    peak_index = int(np.argmax(projection))
    return float(peak_index) / float(len(projection) - 1)