
import cv2
import numpy as np


def load_image(image_path):
    """Load the input handwritten digit image."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def convert_to_grayscale(image):
    """
    Convert the image into grayscale using the best channel for
    foreground/background separation.

    For colored images (e.g. blue handwriting on light background), standard
    BGR2GRAY may not separate colored strokes from background well. This
    function tries multiple conversion methods and returns the one that
    maximizes contrast between foreground and background.

    Parameters
    ----------
    image : numpy.ndarray
        Input color image (BGR).

    Returns
    -------
    numpy.ndarray
        Grayscale image.
    """

    # Standard grayscale conversion.
    gray_standard = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If the image is already grayscale (all channels equal), return directly.
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if np.allclose(b, g) and np.allclose(g, r):
        return gray_standard

    # For colored images, try individual channels and luminance-based
    # conversions to find the best foreground/background separation.
    candidates = {
        "standard": gray_standard,
        "blue_channel": b,
        "green_channel": g,
        "red_channel": r,
        "blue_minus_red": np.clip(b.astype(int) - r.astype(int), 0, 255).astype(np.uint8),
        "blue_minus_green": np.clip(b.astype(int) - g.astype(int), 0, 255).astype(np.uint8),
    }

    best_gray = gray_standard
    best_score = -1.0

    for name, candidate in candidates.items():
        # Otsu threshold to get binary, then measure foreground/background
        # separation by the between-class variance (Otsu's criterion).
        _, binary = cv2.threshold(
            candidate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        fg_pixels = np.count_nonzero(binary == 255)
        total = binary.size
        fg_ratio = fg_pixels / total

        # Ideal foreground ratio for a digit is roughly 5-40%. Score based on
        # how close we are to that range, penalizing too little or too much.
        if 0.05 <= fg_ratio <= 0.40:
            score = 1.0 - abs(fg_ratio - 0.20) / 0.20
        elif fg_ratio < 0.05:
            score = fg_ratio / 0.05
        else:
            score = max(0.0, 1.0 - (fg_ratio - 0.40) / 0.60)

        if score > best_score:
            best_score = score
            best_gray = candidate

    return best_gray


def remove_noise(gray):
    """Apply Gaussian blur to reduce noise."""

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return blurred


def apply_threshold(blurred):
    """Convert grayscale image into binary image using Otsu thresholding."""

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary


def apply_adaptive_threshold(gray):
    """Convert grayscale image using adaptive thresholding.

    Better for textured or unevenly-lit images where a single global
    threshold (Otsu) may not capture local foreground/background
    differences.
    """

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4
    )

    return binary


def try_preprocessing_variants(image):
    """Try multiple preprocessing variants and return the best binary image.

    Variants:
      1. Standard grayscale + Otsu threshold
      2. Best color-channel grayscale + Otsu threshold
      3. Standard grayscale + adaptive threshold
      4. Best color-channel grayscale + adaptive threshold

    Returns the variant that produces a binary image with a reasonable
    foreground ratio (5-40%) and the most digit-like connected component
    structure.
    """

    variants = []

    # Variant 1: Standard grayscale + Otsu
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    binary1 = apply_threshold(blurred1)
    variants.append(("standard_otsu", binary1))

    # Variant 2: Best channel grayscale + Otsu
    gray2 = convert_to_grayscale(image)
    blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    binary2 = apply_threshold(blurred2)
    variants.append(("channel_otsu", binary2))

    # Variant 3: Standard grayscale + adaptive threshold
    gray3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary3 = apply_adaptive_threshold(gray3)
    variants.append(("standard_adaptive", binary3))

    # Variant 4: Best channel grayscale + adaptive threshold
    gray4 = convert_to_grayscale(image)
    binary4 = apply_adaptive_threshold(gray4)
    variants.append(("channel_adaptive", binary4))

    # Score each variant.
    best_variant = None
    best_score = -1.0

    for name, binary in variants:
        score = _score_binary_variant(binary)
        if score > best_score:
            best_score = score
            best_variant = (name, binary)

    return best_variant


def _score_binary_variant(binary):
    """Score a binary image variant for digit-likeness.

    Criteria:
      - Foreground ratio should be reasonable (not too sparse, not too dense).
      - Largest connected component should be substantial.
      - Component should have a reasonable aspect ratio.
    """

    if binary is None or binary.size == 0:
        return -1.0

    fg_pixels = np.count_nonzero(binary == 255)
    total = binary.size
    fg_ratio = fg_pixels / total

    # Foreground ratio score: ideal is 5-40%.
    if fg_ratio < 0.01 or fg_ratio > 0.80:
        return -1.0

    if 0.05 <= fg_ratio <= 0.40:
        ratio_score = 1.0 - abs(fg_ratio - 0.20) / 0.20
    elif fg_ratio < 0.05:
        ratio_score = fg_ratio / 0.05
    else:
        ratio_score = max(0.0, 1.0 - (fg_ratio - 0.40) / 0.40)

    # Connected component analysis.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return ratio_score * 0.5  # No components, poor score

    # Find the largest component.
    largest_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area

    # Largest component should be a significant fraction of foreground.
    if fg_pixels > 0:
        largest_fraction = largest_area / fg_pixels
    else:
        largest_fraction = 0.0

    # Score: prefer variants where the largest component dominates.
    component_score = min(1.0, largest_fraction * 2.0)

    return ratio_score * 0.6 + component_score * 0.4


def remove_small_components(binary_image, min_area=50):
    """
    Remove small noise blobs from a binary image using connected component
    analysis.

    Only isolated foreground components smaller than min_area are removed.
    Internal imperfections (holes enclosed by the digit, e.g. the interior of
    a ``0``, ``6``, ``8`` or ``9``) are background regions surrounded by
    foreground and are never treated as components, so the digit's shape is
    not aggressively modified.

    Assumptions:
    - The digit (foreground) is white (255) and the background is black (0).
    - The handwritten digit forms the largest connected foreground component.
    - Isolated noise specks are smaller than min_area pixels.
    - A digit broken into several pieces by thresholding is kept piece-wise;
      each piece meeting min_area is preserved.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary image (0 = background, 255 = foreground).
    min_area : int
        Minimum pixel area for a component to be kept. Components below this
        threshold are treated as noise and removed.

    Returns
    -------
    numpy.ndarray
        Binary image with small components removed.

    Raises
    ------
    ValueError
        If no foreground component is detected (image is entirely background).
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    if num_labels <= 1:
        raise ValueError(
            "No digit detected: no foreground component found in the "
            "binary image."
        )

    cleaned = np.zeros_like(binary_image)

    # Track the largest component so the main digit is always preserved,
    # even defensively if it falls below min_area.
    largest_label = 0
    largest_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area or label == largest_label:
            cleaned[labels == label] = 255

    return cleaned


def select_best_component(binary_image, min_area=100):
    """
    Select the best connected component that represents a handwritten digit,
    filtering out watermarks, text, and other artifacts.

    Instead of blindly taking the largest component, this function evaluates
    each component based on:
      - Size (should be substantial but not the entire image)
      - Aspect ratio (should be within digit-like range)
      - Position (should be reasonably centered)
      - Solidity (filled fraction of bounding box)
      - Density relative to image size

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary image (0 = background, 255 = foreground).
    min_area : int
        Minimum pixel area for a component to be considered.

    Returns
    -------
    tuple of (numpy.ndarray, dict or None)
        cropped : binary image cropped to the best component's bounding box
                  (with padding), or None if no suitable component found.
        info : dict with bounding box and metadata, or None.
    """

    if binary_image is None or binary_image.size == 0:
        return None, None

    height, width = binary_image.shape
    total_pixels = height * width

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    if num_labels <= 1:
        return None, None

    candidates = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # Skip components that are too small.
        if area < min_area:
            continue

        # Skip components that are too large (likely background artifacts).
        if area > total_pixels * 0.80:
            continue

        # Compute aspect ratio.
        aspect_ratio = w / h if h > 0 else 0.0

        # Skip extremely thin or wide components (unlikely to be a digit).
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            continue

        # Compute component density (filled fraction of bounding box).
        bbox_area = w * h
        density = area / bbox_area if bbox_area > 0 else 0.0

        # Skip components that are too sparse or too solid.
        if density < 0.05 or density > 0.95:
            continue

        # Compute centroid position (normalized).
        cx = centroids[label][0] / width if width > 0 else 0.5
        cy = centroids[label][1] / height if height > 0 else 0.5

        # Position score: prefer components near the center.
        pos_score = 1.0 - abs(cx - 0.5) - abs(cy - 0.5)
        pos_score = max(0.0, pos_score)

        # Size score: prefer components that are a reasonable fraction of
        # the image (not tiny specks, not dominating the image).
        size_fraction = area / total_pixels
        if 0.005 <= size_fraction <= 0.40:
            size_score = 1.0
        elif size_fraction < 0.005:
            size_score = size_fraction / 0.005
        else:
            size_score = max(0.0, 1.0 - (size_fraction - 0.40) / 0.40)

        # Combined score.
        score = size_score * 0.4 + pos_score * 0.3 + density * 0.2 + min(1.0, aspect_ratio / 2.0) * 0.1

        candidates.append({
            "label": label,
            "area": area,
            "bbox": (int(x), int(y), int(w), int(h)),
            "aspect_ratio": aspect_ratio,
            "density": density,
            "score": score,
            "centroid": (float(cx), float(cy)),
        })

    if not candidates:
        return None, None

    # Sort by score descending.
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    x, y, w, h = best["bbox"]

    # Apply padding.
    padding = 10
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(width, x + w + padding)
    y_end = min(height, y + h + padding)

    cropped = binary_image[y_start:y_end, x_start:x_end]

    info = {
        "bbox": (int(x), int(y), int(w), int(h)),
        "touches_boundary": bool(
            x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height
        ),
        "padded_bbox": (
            int(x_start),
            int(y_start),
            int(x_end - x_start),
            int(y_end - y_start),
        ),
        "component_score": float(best["score"]),
        "component_area": int(best["area"]),
        "component_density": float(best["density"]),
        "num_candidates": len(candidates),
    }

    return cropped, info


def crop_digit(binary_image, padding=10):
    """
    Crop the cleaned binary image to isolate the handwritten digit.

    Uses select_best_component() to find the most digit-like connected
    component, filtering out watermarks, text, and other artifacts.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Cleaned binary image (0 = background, 255 = foreground).
    padding : int
        Number of pixels of padding to add around the cropped digit on each
        side. Must be non-negative.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
        cropped : the cropped (and padded) binary digit image.
        info : dict with keys:
            - "bbox" : (x, y, w, h) bounding box in the original image
            - "touches_boundary" : bool, True if the digit touches an image
              edge (may be clipped)
            - "padded_bbox" : (x, y, w, h) after padding applied
            - "component_score" : float, quality score of selected component
            - "component_area" : int, pixel area of selected component
            - "component_density" : float, filled fraction of bounding box
            - "num_candidates" : int, number of candidate components found

    Raises
    ------
    ValueError
        If padding is negative or no foreground digit is detected.
    """

    if padding < 0:
        raise ValueError("padding must be non-negative")

    cropped, info = select_best_component(binary_image, min_area=50)

    if cropped is None:
        raise ValueError(
            "No digit detected: no suitable foreground component found in "
            "the binary image."
        )

    return cropped, info


def _significant_enclosed_hole_count(binary_image, minimum_area_ratio=0.08):
    """Count substantial enclosed regions, ignoring small pencil gaps."""
    points = cv2.findNonZero(binary_image)
    if points is None:
        return 0

    x, y, width, height = cv2.boundingRect(points)
    digit = binary_image[y:y + height, x:x + width]
    contours, hierarchy = cv2.findContours(
        digit.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0

    minimum_area = float(width * height) * minimum_area_ratio
    holes = 0
    for index, contour in enumerate(contours):
        depth = 0
        parent = int(hierarchy[0][index][3])
        while parent != -1:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        if depth % 2 == 1 and cv2.contourArea(contour) >= minimum_area:
            holes += 1
    return holes


def normalize_digit_image(cropped_binary):
    """
    Normalize a cropped binary digit image to a standardized 28x28 canvas.

    Finds the digit's bounding box within the cropped image (ignoring any
    padding), resizes the digit proportionally so it fits inside a 20x20
    area while preserving its aspect ratio, then centers it on a 28x28 black
    canvas. The digit remains white (255) and the background remains black (0).

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Cropped binary digit image (0 = background, 255 = foreground).

    Returns
    -------
    numpy.ndarray
        Normalized 28x28 binary image with the digit centered.

    Raises
    ------
    TypeError
        If the input is not a NumPy array.
    ValueError
        If the input image is empty, not 2D, or contains no foreground pixels.
    """

    # Validate input type.
    if not isinstance(cropped_binary, np.ndarray):
        raise TypeError(
            "cropped_binary must be a NumPy array."
        )

    # Validate that the image is not empty.
    if cropped_binary.size == 0:
        raise ValueError(
            "cropped_binary is empty: cannot normalize an empty image."
        )

    # Ensure the image is 2D (single-channel binary).
    if cropped_binary.ndim != 2:
        raise ValueError(
            "cropped_binary must be a 2D single-channel image."
        )

    # Validate that the image contains foreground pixels.
    foreground_count = int(np.count_nonzero(cropped_binary == 255))
    if foreground_count == 0:
        raise ValueError(
            "cropped_binary contains no foreground pixels: cannot normalize "
            "an image without a digit."
        )

    # Compute digit bounding box and validate digit-likeness before
    # normalization.
    contours, _ = cv2.findContours(
        cropped_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "No digit contour found: cannot normalize an image without a "
            "detectable digit."
        )

    # Largest contour is the presumed digit.
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, digit_w, digit_h = cv2.boundingRect(largest_contour)

    digit_area = float(cv2.contourArea(largest_contour))
    bbox_area = float(digit_w * digit_h)
    digit_density_in_bbox = digit_area / bbox_area if bbox_area > 0 else 0.0

    aspect_ratio = digit_w / digit_h if digit_h > 0 else 0.0

    # Reject extreme aspect ratios that cannot be a single handwritten digit.
    if aspect_ratio < 0.20 or aspect_ratio > 5.0:
        raise ValueError(
            "Digit aspect ratio {:.2f} is outside the valid range "
            "[0.20, 5.0]. Input may contain multiple objects or a "
            "non-digit shape.".format(aspect_ratio)
        )

    # Reject if the digit occupies almost none of its bounding box
    # (scattered noise) or almost all of it (solid block).
    if digit_density_in_bbox < 0.05:
        raise ValueError(
            "Digit occupies less than 5% of its bounding box. Input may "
            "contain scattered noise rather than a single digit."
        )
    if digit_density_in_bbox > 0.95:
        raise ValueError(
            "Digit occupies more than 95% of its bounding box. Input may "
            "be a solid block rather than a handwritten digit."
        )

    # Extract just the digit region (no padding).
    digit_region = cropped_binary[y:y + digit_h, x:x + digit_w]

    # Target canvas size.
    canvas_size = 28
    inner_size = 20  # digit fits inside this area

    # Compute scaling factor to fit the digit inside inner_size x inner_size
    # while preserving aspect ratio.
    scale = min(inner_size / digit_w, inner_size / digit_h)

    # Compute new dimensions after proportional scaling.
    new_width = max(1, int(round(digit_w * scale)))
    new_height = max(1, int(round(digit_h * scale)))

    # Resize the digit proportionally.
    # Keep the existing nearest-neighbor result unless it loses a substantial
    # enclosed region. Area interpolation can preserve narrow loops in large,
    # naturally written digits that otherwise open during downscaling.
    resized = cv2.resize(
        digit_region,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )
    _, resized = cv2.threshold(
        resized, 127, 255, cv2.THRESH_BINARY
    )

    source_holes = _significant_enclosed_hole_count(digit_region)
    if source_holes > _significant_enclosed_hole_count(resized):
        area_resized = cv2.resize(
            digit_region,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
        _, area_resized = cv2.threshold(
            area_resized, 80, 255, cv2.THRESH_BINARY
        )
        if _significant_enclosed_hole_count(area_resized) >= source_holes:
            resized = area_resized

    # Create a 28x28 black canvas.
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Compute top-left offset to center the resized digit on the canvas.
    x_offset = (canvas_size - new_width) // 2
    y_offset = (canvas_size - new_height) // 2

    # Place the resized digit at the center of the canvas.
    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return canvas


def preprocess_image(image_path, min_area=50):
    """
    Run the full preprocessing pipeline on an image, trying multiple
    variants and selecting the best result.

    Parameters
    ----------
    image_path : str
        Path to the input handwritten digit image.
    min_area : int
        Minimum pixel area for a foreground component to be kept. Components
        below this threshold are treated as noise and removed.

    Returns
    -------
    tuple of numpy.ndarray
        (original, grayscale, blurred, binary, cleaned).

    Raises
    ------
    FileNotFoundError
        If the image file cannot be loaded.
    ValueError
        If no digit is detected after thresholding.
    """

    image = load_image(image_path)

    # Try multiple preprocessing variants.
    variant_name, binary = try_preprocessing_variants(image)

    # Also compute grayscale and blurred for return compatibility.
    gray = convert_to_grayscale(image)
    blurred = remove_noise(gray)

    # Clean with morphological closing.
    cleaned = clean_binary(binary)

    # Remove small components.
    cleaned = remove_small_components(cleaned, min_area=min_area)

    return image, gray, blurred, binary, cleaned


def clean_binary(binary_image):
    """
    Fill small gaps and holes inside the digit.
    """

    kernel = np.ones((3, 3), np.uint8)

    cleaned = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return cleaned
