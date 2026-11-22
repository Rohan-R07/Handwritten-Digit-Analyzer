
import numpy as np


def _safe_get(features, key, default=None):
    """Safely retrieve a value from the features dictionary."""
    if features is None:
        return default
    return features.get(key, default)


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


def classify_features(normalized_features):
    """
    Rule-based digit classification using normalized geometric features.

    This is a preliminary heuristic classifier. All thresholds and scoring
    weights are unvalidated starting points intended for experimentation,
    not production use.

    Parameters
    ----------
    normalized_features : dict
        Dictionary from extract_normalized_features() with keys:
            - aspect_ratio
            - foreground_density
            - normalized_horizontal_projection
            - normalized_vertical_projection
            - normalized_contour_area
            - normalized_hole_areas
            - hole_count

    Returns
    -------
    dict
        {
            "scores": {digit: score_float},        # scores for digits 0-9
            "ranked_candidates": [(digit, score), ...],  # descending score
            "predicted_digit": int or None,
            "best_score": float or None,
            "rule_observations": {digit: [observation_strings]},
            "feature_summary": {key: value},
            "status": "ok" | "insufficient_features" | "error",
            "error_message": str or None
        }
    """

    result = {
        "scores": {},
        "ranked_candidates": [],
        "predicted_digit": None,
        "best_score": None,
        "rule_observations": {},
        "feature_summary": {},
        "status": "error",
        "error_message": None,
    }

    # --- Validate input ---
    if normalized_features is None:
        result["error_message"] = "normalized_features is None."
        return result

    if not isinstance(normalized_features, dict):
        result["error_message"] = "normalized_features must be a dict."
        return result

    # --- Extract features with safe defaults ---
    aspect_ratio = _safe_get(normalized_features, "aspect_ratio")
    foreground_density = _safe_get(normalized_features, "foreground_density")
    norm_h_projection = _safe_get(normalized_features, "normalized_horizontal_projection")
    norm_v_projection = _safe_get(normalized_features, "normalized_vertical_projection")
    norm_contour_area = _safe_get(normalized_features, "normalized_contour_area")
    normalized_hole_areas = _safe_get(normalized_features, "normalized_hole_areas", [])
    hole_count = _safe_get(normalized_features, "hole_count")

    # --- Check for insufficient features ---
    critical_missing = []
    if aspect_ratio is None:
        critical_missing.append("aspect_ratio")
    if foreground_density is None:
        critical_missing.append("foreground_density")
    if hole_count is None:
        critical_missing.append("hole_count")

    if critical_missing:
        result["status"] = "insufficient_features"
        result["error_message"] = (
            "Missing critical features: {}".format(", ".join(critical_missing))
        )
        # Still return zero scores so the structure is consistent
        for d in range(10):
            result["scores"][d] = 0.0
            result["rule_observations"][d] = [
                "Insufficient features for classification."
            ]
        result["ranked_candidates"] = [(d, 0.0) for d in range(10)]
        return result

    # --- Feature summary for reporting ---
    result["feature_summary"] = {
        "aspect_ratio": float(aspect_ratio),
        "foreground_density": float(foreground_density),
        "norm_contour_area": float(norm_contour_area) if norm_contour_area is not None else None,
        "hole_count": int(hole_count),
        "num_hole_areas": len(normalized_hole_areas) if normalized_hole_areas else 0,
        "h_projection_peak_pos": _projection_peak_position(norm_h_projection, "horizontal"),
        "v_projection_peak_pos": _projection_peak_position(norm_v_projection, "vertical"),
        "h_projection_top_heavy": _projection_top_heavy_ratio(norm_h_projection),
        "v_projection_left_heavy": _projection_left_heavy_ratio(norm_v_projection),
    }

    # --- Derived feature heuristics ---
    is_tall = aspect_ratio < 0.65       # height clearly greater than width
    is_wide = aspect_ratio > 1.25       # width > height significantly
    is_squareish = 0.75 <= aspect_ratio <= 1.25
    is_sparse = foreground_density < 0.12   # very thin stroke
    is_dense = foreground_density > 0.30
    has_holes = hole_count > 0
    has_one_hole = hole_count == 1
    has_two_holes = hole_count == 2
    no_holes = hole_count == 0

    h_peak_pos = result["feature_summary"]["h_projection_peak_pos"]
    v_peak_pos = result["feature_summary"]["v_projection_peak_pos"]
    h_top_heavy = result["feature_summary"]["h_projection_top_heavy"]
    v_left_heavy = result["feature_summary"]["v_projection_left_heavy"]

    # Detect rising pattern at top of digit (curve going up from top edge).
    # Digit 2 has a curve at top (rising projection), while digit 7 has a
    # flat top bar. This helps distinguish them.
    rising_at_top = False
    if norm_h_projection is not None and len(norm_h_projection) > 10:
        h_arr = np.asarray(norm_h_projection, dtype=np.float64)
        non_zero = np.nonzero(h_arr)[0]
        if len(non_zero) > 5:
            d_top = non_zero[0]
            d_mid = (d_top + len(non_zero) - 1) // 2
            top_region = h_arr[d_top:d_mid]
            if len(top_region) > 3:
                first_quarter = np.mean(top_region[:len(top_region) // 4])
                last_quarter = np.mean(top_region[len(top_region) * 3 // 4:])
                if last_quarter > first_quarter * 1.5 and last_quarter > 0.15:
                    rising_at_top = True

    # Hole position data for digit 8 detection.
    normalized_hole_positions = _safe_get(
        normalized_features, "normalized_hole_positions", []
    )
    normalized_hole_areas = _safe_get(
        normalized_features, "normalized_hole_areas", []
    )

    # Determine if holes are vertically arranged (one upper, one lower).
    # Handles both exactly 2 holes and 3 holes where the middle one is a
    # small gap between two loops (common in handwritten 8s).
    holes_vertically_arranged = False
    hole_upper_center_y = None
    hole_lower_center_y = None

    if len(normalized_hole_positions) >= 2:
        # Sort holes by vertical position.
        sorted_holes = sorted(
            zip(normalized_hole_positions, normalized_hole_areas),
            key=lambda x: x[0][1]
        )

        # Check if the topmost and bottommost holes are vertically separated.
        top_hole = sorted_holes[0]
        bottom_hole = sorted_holes[-1]
        top_y = top_hole[0][1]
        bottom_y = bottom_hole[0][1]

        # One hole in upper region, one in lower region.
        if top_y < 0.45 and bottom_y > 0.55:
            holes_vertically_arranged = True
            hole_upper_center_y = top_y
            hole_lower_center_y = bottom_y

    # Determine if hole areas are balanced (similar sizes).
    # Uses the two largest holes when there are 3+ holes.
    hole_areas_balanced = False
    if len(normalized_hole_areas) >= 2:
        # Take the two largest hole areas.
        sorted_areas = sorted(normalized_hole_areas, reverse=True)
        a1, a2 = sorted_areas[0], sorted_areas[1]
        if a1 > 0 and a2 > 0:
            ratio = min(a1, a2) / max(a1, a2)
            hole_areas_balanced = ratio > 0.4  # areas within 2.5x of each other

    # Determine if there are exactly 2 significant holes (or 2 large + small gap).
    two_significant_holes = False
    if len(normalized_hole_areas) == 2:
        two_significant_holes = True
    elif len(normalized_hole_areas) >= 3:
        # Check if the two largest holes are vertically arranged and the
        # remaining holes are small (likely gaps between loops).
        sorted_areas = sorted(normalized_hole_areas, reverse=True)
        # Two largest holes dominate; third hole is small relative to them.
        if sorted_areas[1] > 0 and sorted_areas[2] < 0.5 * sorted_areas[1]:
            # Also verify the two largest holes are vertically arranged.
            y_positions = [pos[1] for pos in normalized_hole_positions]
            y_positions.sort()
            if y_positions[0] < 0.45 and y_positions[-1] > 0.55:
                two_significant_holes = True

    # --- Score each digit ---
    scores = {}
    observations = {}

    # ---- Digit 0: oval loop, 1 centered hole, dense, square-ish ----
    score_0 = 0.0
    obs_0 = []
    if has_one_hole:
        score_0 += 0.45
        obs_0.append("1 enclosed hole (consistent with 0)")
    else:
        obs_0.append("hole count != 1 (weakens 0)")

    if is_squareish:
        score_0 += 0.20
        obs_0.append("aspect ratio near 1 (consistent with 0)")
    else:
        obs_0.append("aspect ratio not near 1 (weakens 0)")

    if is_dense:
        score_0 += 0.20
        obs_0.append("high density (consistent with 0)")
    elif foreground_density > 0.15:
        score_0 += 0.10
        obs_0.append("moderate density")

    # Hole centered horizontally and vertically
    if abs(h_peak_pos - 0.5) < 0.2 and abs(v_peak_pos - 0.5) < 0.2:
        score_0 += 0.20
        obs_0.append("projection peak centered (consistent with 0)")
    elif abs(h_peak_pos - 0.5) < 0.3 and abs(v_peak_pos - 0.5) < 0.3:
        score_0 += 0.10
        obs_0.append("projection roughly centered")

    scores[0] = min(score_0, 1.0)
    observations[0] = obs_0

    # ---- Digit 1: thin vertical stroke, tall, sparse, no holes ----
    score_1 = 0.0
    obs_1 = []
    if no_holes:
        score_1 += 0.15
        obs_1.append("no holes (consistent with 1)")
    else:
        obs_1.append("has holes (strongly inconsistent with 1)")

    # Digit 1 must be clearly tall (height much greater than width).
    if is_tall:
        score_1 += 0.25
        obs_1.append("tall aspect ratio (consistent with 1)")
    elif is_squareish:
        obs_1.append("aspect ratio not tall enough for 1 (weakens 1)")
    else:
        obs_1.append("wide aspect ratio (inconsistent with 1)")

    # Digit 1 is a thin stroke: density must be low.
    if is_sparse:
        score_1 += 0.20
        obs_1.append("low density (consistent with thin stroke of 1)")
    elif foreground_density < 0.20:
        score_1 += 0.05
        obs_1.append("moderate-low density (weakly consistent with 1)")
    else:
        obs_1.append("density too high for 1 (weakens 1)")

    if abs(v_peak_pos - 0.5) < 0.15:
        score_1 += 0.15
        obs_1.append("vertical projection centered (consistent with 1)")
    else:
        obs_1.append("vertical projection off-center (weakens 1)")

    scores[1] = min(score_1, 1.0)
    observations[1] = obs_1

    # ---- Digit 2: curve-top, diagonal, base, no holes ----
    score_2 = 0.0
    obs_2 = []
    if no_holes:
        score_2 += 0.20
        obs_2.append("no holes (consistent with 2)")
    else:
        obs_2.append("has holes (inconsistent with 2)")

    if foreground_density > 0.10 and foreground_density < 0.35:
        score_2 += 0.15
        obs_2.append("moderate density (consistent with 2)")

    if is_squareish:
        score_2 += 0.15
        obs_2.append("aspect ratio near 1 (consistent with 2)")

    # Top-heavy horizontal projection (curve at top).
    # Compute within the digit's non-empty vertical range to handle cases
    # where the digit is not centered vertically in the canvas.
    h_top_heavy_digit = h_top_heavy
    # Detect rising pattern at top of digit (curve going up from top edge).
    rising_at_top = False
    if norm_h_projection is not None and len(norm_h_projection) > 10:
        h_arr = np.asarray(norm_h_projection, dtype=np.float64)
        non_zero = np.nonzero(h_arr)[0]
        if len(non_zero) > 5:
            d_top = non_zero[0]
            d_bottom = non_zero[-1]
            d_mid = (d_top + d_bottom) // 2
            top_half = np.sum(h_arr[d_top:d_mid])
            bottom_half = np.sum(h_arr[d_mid:d_bottom + 1])
            if bottom_half > 0:
                h_top_heavy_digit = float(top_half) / float(bottom_half)
            # Detect rising pattern: projection increases from top of digit
            # to somewhere in the upper half (curve going up, not flat bar).
            top_region = h_arr[d_top:d_mid]
            if len(top_region) > 3:
                # Check if projection generally increases from top to mid
                first_quarter = np.mean(top_region[:len(top_region)//4])
                last_quarter = np.mean(top_region[len(top_region)*3//4:])
                if last_quarter > first_quarter * 1.5 and last_quarter > 0.15:
                    rising_at_top = True

    if h_top_heavy_digit > 1.05 or rising_at_top:
        # Digit 2 has a curve at top (rising projection). But digit 3 also
        # has curves at top and bottom with a right-heavy vertical projection
        # throughout. If the vertical projection is consistently right-heavy,
        # it's more likely digit 3 than digit 2.
        if not (v_left_heavy < 0.85):
            score_2 += 0.35
            obs_2.append("curved top boundary (consistent with 2)")
        else:
            score_2 += 0.05
            obs_2.append("some curvature at top (weak evidence for 2)")
    elif h_top_heavy_digit > 0.95:
        score_2 += 0.10
        obs_2.append("balanced horizontal projection")

    # Check for horizontal projection valley in middle (curve-diagonal-base
    # pattern: heavy top, lighter middle, heavy bottom).
    if norm_h_projection is not None and len(norm_h_projection) > 10:
        h_arr = np.asarray(norm_h_projection, dtype=np.float64)
        non_zero = np.nonzero(h_arr)[0]
        if len(non_zero) > 10:
            d_top = non_zero[0]
            d_bottom = non_zero[-1]
            d_mid = (d_top + d_bottom) // 2
            top_mass = np.sum(h_arr[d_top:d_mid])
            middle_mass = np.sum(h_arr[d_mid:d_bottom + 1])
            # Split middle into upper-middle and lower-middle
            mid2 = (d_mid + d_bottom) // 2
            mid_mass = np.sum(h_arr[d_mid:mid2])
            bottom_mass = np.sum(h_arr[mid2:d_bottom + 1])
            total = top_mass + mid_mass + bottom_mass
            if total > 0:
                top_fraction = top_mass / total
                mid_fraction = mid_mass / total
                bottom_fraction = bottom_mass / total
                # Digit 2 has heavy top (curve), lighter middle (diagonal),
                # heavy bottom (base).
                if top_fraction > 0.30 and bottom_fraction > 0.25 and mid_fraction < 0.45:
                    score_2 += 0.15
                    obs_2.append("curve-diagonal-base projection pattern (consistent with 2)")

    scores[2] = min(score_2, 1.0)
    observations[2] = obs_2

    # ---- Digit 3: two right-side curves, no holes ----
    score_3 = 0.0
    obs_3 = []
    if no_holes:
        score_3 += 0.20
        obs_3.append("no holes (consistent with 3)")
    else:
        obs_3.append("has holes (inconsistent with 3)")

    if foreground_density > 0.15 and foreground_density < 0.35:
        score_3 += 0.15
        obs_3.append("moderate density (consistent with 3)")

    if is_squareish:
        score_3 += 0.10
        obs_3.append("aspect ratio near 1 (consistent with 3)")

    # Right-heavy vertical projection (curves on right).
    if v_left_heavy < 0.85:
        score_3 += 0.25
        obs_3.append("right-heavy vertical projection (consistent with 3)")
    elif v_left_heavy > 1.15:
        score_3 += 0.05
        obs_3.append("left-heavy projection (weakens 3)")

    # Check for two peaks in horizontal projection (two curved lobes).
    if norm_h_projection is not None and len(norm_h_projection) > 10:
        h_arr = np.asarray(norm_h_projection, dtype=np.float64)
        # Smooth slightly to find peaks.
        if len(h_arr) > 5:
            kernel = np.ones(3) / 3.0
            h_smooth = np.convolve(h_arr, kernel, mode='same')
            # Find local maxima.
            peaks = []
            for i in range(1, len(h_smooth) - 1):
                if h_smooth[i] >= h_smooth[i - 1] and h_smooth[i] >= h_smooth[i + 1]:
                    peaks.append((i, h_smooth[i]))
            # Sort by height descending.
            peaks.sort(key=lambda x: x[1], reverse=True)
            # Digit 3 has two distinct peaks (upper and lower lobes).
            if len(peaks) >= 2:
                peak1_pos = peaks[0][0] / len(h_smooth)
                peak2_pos = peaks[1][0] / len(h_smooth)
                peak_separation = abs(peak1_pos - peak2_pos)
                # Peaks should be in upper and lower halves, separated.
                if peak_separation > 0.25:
                    score_3 += 0.25
                    obs_3.append("two curved lobes in projection (consistent with 3)")
                # Additional bonus: right-heavy vertical projection + two peaks
                # strongly indicates digit 3 (curves on right side).
                if peak_separation > 0.25 and v_left_heavy < 0.90:
                    score_3 += 0.10
                    obs_3.append("right-side curved lobes (strongly consistent with 3)")

    scores[3] = min(score_3, 1.0)
    observations[3] = obs_3

    # ---- Digit 4: open top, vertical right stroke, no holes ----
    score_4 = 0.0
    obs_4 = []
    if no_holes:
        score_4 += 0.25
        obs_4.append("no holes (consistent with 4)")
    else:
        obs_4.append("has holes (inconsistent with 4)")

    if foreground_density > 0.10 and foreground_density < 0.30:
        score_4 += 0.20
        obs_4.append("moderate-low density (consistent with 4)")

    # Right-heavy vertical projection (vertical stroke on right)
    if v_left_heavy < 0.85:
        score_4 += 0.25
        obs_4.append("right-heavy vertical projection (consistent with 4)")
    elif v_left_heavy > 1.15:
        score_4 += 0.05
        obs_4.append("left-heavy projection (weakens 4)")

    scores[4] = min(score_4, 1.0)
    observations[4] = obs_4

    # ---- Digit 5: curve top, horizontal base, no holes ----
    score_5 = 0.0
    obs_5 = []
    if no_holes:
        score_5 += 0.25
        obs_5.append("no holes (consistent with 5)")
    else:
        obs_5.append("has holes (inconsistent with 5)")

    if foreground_density > 0.15 and foreground_density < 0.35:
        score_5 += 0.20
        obs_5.append("moderate density (consistent with 5)")

    if is_squareish:
        score_5 += 0.15
        obs_5.append("aspect ratio near 1 (consistent with 5)")

    # Bottom-heavy horizontal projection (base at bottom)
    if h_top_heavy < 0.8:
        score_5 += 0.20
        obs_5.append("bottom-heavy horizontal projection (consistent with 5)")
    elif h_top_heavy > 1.2:
        score_5 += 0.05
        obs_5.append("top-heavy projection (weakens 5)")

    scores[5] = min(score_5, 1.0)
    observations[5] = obs_5

    # ---- Digit 6: loop at bottom, 1 hole at bottom ----
    score_6 = 0.0
    obs_6 = []
    if has_one_hole:
        score_6 += 0.35
        obs_6.append("1 enclosed hole (consistent with 6)")
    else:
        obs_6.append("hole count != 1 (weakens 6)")

    if foreground_density > 0.20:
        score_6 += 0.20
        obs_6.append("moderate-high density (consistent with 6)")

    if is_squareish:
        score_6 += 0.15
        obs_6.append("aspect ratio near 1 (consistent with 6)")

    # Bottom-heavy projection (loop at bottom)
    if h_top_heavy < 0.8:
        score_6 += 0.30
        obs_6.append("bottom-heavy projection (consistent with 6)")
    elif h_top_heavy > 1.2:
        score_6 += 0.05
        obs_6.append("top-heavy projection (weakens 6)")

    scores[6] = min(score_6, 1.0)
    observations[6] = obs_6

    # ---- Digit 7: top bar + diagonal, no holes, tall/sparse ----
    score_7 = 0.0
    obs_7 = []
    if no_holes:
        score_7 += 0.25
        obs_7.append("no holes (consistent with 7)")
    else:
        obs_7.append("has holes (inconsistent with 7)")

    if is_tall or is_squareish:
        score_7 += 0.20
        obs_7.append("aspect ratio consistent with 7")

    if is_sparse:
        score_7 += 0.25
        obs_7.append("low density (consistent with thin stroke of 7)")

    # Top-heavy horizontal projection (bar at top)
    if h_top_heavy > 1.2:
        score_7 += 0.30
        obs_7.append("top-heavy horizontal projection (consistent with 7)")
    elif h_top_heavy > 0.9:
        score_7 += 0.10
        obs_7.append("balanced horizontal projection")

    # Rising curve at top (inconsistent with digit 7's flat top bar)
    if rising_at_top:
        score_7 -= 0.25
        obs_7.append("rising curve at top (inconsistent with 7)")

    scores[7] = min(score_7, 1.0)
    observations[7] = obs_7

    # ---- Digit 8: two loops, 2 holes, very dense ----
    score_8 = 0.0
    obs_8 = []
    if two_significant_holes:
        score_8 += 0.40
        obs_8.append("2 enclosed holes (consistent with 8)")
        if holes_vertically_arranged:
            score_8 += 0.25
            obs_8.append("holes vertically stacked (strongly consistent with 8)")
        else:
            obs_8.append("holes not vertically stacked (weakens 8)")
        if hole_areas_balanced:
            score_8 += 0.15
            obs_8.append("balanced hole areas (consistent with 8)")
        else:
            obs_8.append("unbalanced hole areas (weakens 8)")
    elif has_one_hole:
        score_8 += 0.05
        obs_8.append("1 hole (weakens 8)")
    else:
        obs_8.append("no holes (inconsistent with 8)")

    if is_dense:
        score_8 += 0.15
        obs_8.append("high density (consistent with 8)")
    elif foreground_density > 0.20:
        score_8 += 0.08
        obs_8.append("moderate-high density")

    if is_squareish:
        score_8 += 0.10
        obs_8.append("aspect ratio near 1 (consistent with 8)")

    # Balanced projections (two stacked loops)
    if 0.8 < h_top_heavy < 1.2:
        score_8 += 0.12
        obs_8.append("balanced horizontal projection (consistent with 8)")

    scores[8] = min(score_8, 1.0)
    observations[8] = obs_8

    # ---- Digit 9: loop at top, 1 hole at top ----
    score_9 = 0.0
    obs_9 = []
    if has_one_hole:
        score_9 += 0.35
        obs_9.append("1 enclosed hole (consistent with 9)")
    else:
        obs_9.append("hole count != 1 (weakens 9)")

    if foreground_density > 0.20:
        score_9 += 0.20
        obs_9.append("moderate-high density (consistent with 9)")

    if is_squareish:
        score_9 += 0.15
        obs_9.append("aspect ratio near 1 (consistent with 9)")

    # Top-heavy projection (loop at top)
    if h_top_heavy > 1.2:
        score_9 += 0.30
        obs_9.append("top-heavy projection (consistent with 9)")
    elif h_top_heavy < 0.8:
        score_9 += 0.05
        obs_9.append("bottom-heavy projection (weakens 9)")

    scores[9] = min(score_9, 1.0)
    observations[9] = obs_9

    # --- Build result ---
    result["scores"] = {d: round(float(scores[d]), 4) for d in range(10)}
    result["rule_observations"] = observations

    # Rank candidates by descending score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    result["ranked_candidates"] = [(int(d), round(float(s), 4)) for d, s in ranked]

    # Predicted digit is the top candidate
    if ranked:
        result["predicted_digit"] = int(ranked[0][0])
        result["best_score"] = round(float(ranked[0][1]), 4)

    result["status"] = "ok"
    result["error_message"] = None

    return result


def explain_classification(classification_result):
    """
    Produce a human-readable explanation of the classification result.

    Parameters
    ----------
    classification_result : dict
        Output from classify_features().

    Returns
    -------
    str
        Formatted explanation text.
    """

    if classification_result is None:
        return "No classification result available."

    if classification_result["status"] != "ok":
        return (
            "Classification status: {}\n"
            "Error: {}"
        ).format(
            classification_result["status"],
            classification_result.get("error_message", "unknown"),
        )

    lines = []
    lines.append("Rule-Based Classification Result")
    lines.append("=" * 40)
    lines.append("Predicted digit: {} (score: {})".format(
        classification_result["predicted_digit"],
        classification_result["best_score"],
    ))
    lines.append("")

    lines.append("Feature Summary:")
    for key, value in classification_result["feature_summary"].items():
        if value is not None:
            if isinstance(value, float):
                lines.append("  {}: {:.4f}".format(key, value))
            else:
                lines.append("  {}: {}".format(key, value))
    lines.append("")

    lines.append("Candidate Scores (descending):")
    for digit, score in classification_result["ranked_candidates"]:
        lines.append("  Digit {}: {:.4f}".format(digit, score))
    lines.append("")

    lines.append("Rule Observations for Top Candidates:")
    top_digits = [d for d, _ in classification_result["ranked_candidates"][:3]]
    for digit in top_digits:
        lines.append("  Digit {}:".format(digit))
        for obs in classification_result["rule_observations"].get(digit, []):
            lines.append("    - {}".format(obs))

    return "\n".join(lines)
