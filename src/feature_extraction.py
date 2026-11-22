
import cv2
import numpy as np


def extract_basic_features(cropped_binary, crop_info=None):
    """
    Extract basic geometric features from a cropped binary digit image.

    Features extracted:
      - Bounding box coordinates (x, y, width, height)
      - Width and height of the digit
      - Aspect ratio (width / height)
      - Foreground pixel count (white pixels)
      - Foreground pixel density (foreground pixels / total pixels)

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Cropped binary image (0 = background, 255 = foreground).
    crop_info : dict or None
        Optional dictionary returned by crop_digit(). If it contains a
        "bbox" key with (x, y, w, h), that bounding box is reused instead of
        recomputing it from the cropped image.

    Returns
    -------
    dict
        Structured dictionary with keys:
            - "bbox" : (x, y, w, h)
            - "width" : int
            - "height" : int
            - "aspect_ratio" : float
            - "foreground_pixel_count" : int
            - "foreground_pixel_density" : float (0.0 - 1.0)

    Raises
    ------
    ValueError
        If the input image is None, empty, or contains no foreground pixels.
    TypeError
        If the input is not a NumPy array.
    """

    # Validate input type.
    if not isinstance(cropped_binary, np.ndarray):
        raise TypeError(
            "cropped_binary must be a NumPy array."
        )

    # Validate that the image is not empty.
    if cropped_binary.size == 0:
        raise ValueError(
            "cropped_binary is empty: cannot extract features from "
            "an empty image."
        )

    # Ensure the image is 2D (grayscale/binary).
    if cropped_binary.ndim != 2:
        raise ValueError(
            "cropped_binary must be a 2D single-channel image."
        )

    height, width = cropped_binary.shape

    # Determine the bounding box.
    if crop_info is not None and "bbox" in crop_info:
        # Reuse the bounding box computed during cropping.
        x, y, w, h = crop_info["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
    else:
        # Compute bounding box from the foreground pixels in the cropped image.
        # findContours locates connected white regions; boundingRect gives the
        # axis-aligned rectangle enclosing the largest one.
        contours, _ = cv2.findContours(
            cropped_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            raise ValueError(
                "No digit detected: no foreground contour found in "
                "the cropped binary image."
            )

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x, y, w, h = int(x), int(y), int(w), int(h)

    # Width and height of the bounding box.
    bbox_width = w
    bbox_height = h

    # Aspect ratio: width divided by height. Guard against zero height.
    if bbox_height == 0:
        raise ValueError(
            "Bounding box height is zero: invalid digit crop."
        )
    aspect_ratio = bbox_width / bbox_height

    # Foreground pixel count: count pixels equal to 255 (white digit).
    foreground_pixel_count = int(np.count_nonzero(cropped_binary == 255))

    # Foreground pixel density: fraction of pixels that are foreground.
    total_pixels = cropped_binary.size
    foreground_pixel_density = foreground_pixel_count / total_pixels

    features = {
        "bbox": (x, y, bbox_width, bbox_height),
        "width": bbox_width,
        "height": bbox_height,
        "aspect_ratio": float(aspect_ratio),
        "foreground_pixel_count": foreground_pixel_count,
        "foreground_pixel_density": float(foreground_pixel_density),
    }

    return features


def extract_projection_features(cropped_binary):
    """
    Extract horizontal and vertical projection profiles from a cropped
    binary digit image.

    Projection profiles count the number of white foreground pixels (255)
    along each row (horizontal) and each column (vertical). These profiles
    describe the distribution of ink across the digit and are useful for
    distinguishing digit shapes.

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Cropped binary image (0 = background, 255 = foreground).

    Returns
    -------
    dict
        Structured dictionary with keys:
            - "horizontal_projection" : numpy.ndarray
                1D array of length = image height. Each entry is the number
                of foreground pixels in that row.
            - "vertical_projection" : numpy.ndarray
                1D array of length = image width. Each entry is the number
                of foreground pixels in that column.
            - "maximum_horizontal_projection" : int
                Maximum value in the horizontal projection (peak row count).
            - "maximum_vertical_projection" : int
                Maximum value in the vertical projection (peak column count).

    Raises
    ------
    TypeError
        If the input is not a NumPy array.
    ValueError
        If the input image is empty or not a 2D single-channel image.
    """

    # Validate input type.
    if not isinstance(cropped_binary, np.ndarray):
        raise TypeError(
            "cropped_binary must be a NumPy array."
        )

    # Validate that the image is not empty.
    if cropped_binary.size == 0:
        raise ValueError(
            "cropped_binary is empty: cannot compute projection profiles "
            "on an empty image."
        )

    # Ensure the image is 2D (single-channel binary).
    if cropped_binary.ndim != 2:
        raise ValueError(
            "cropped_binary must be a 2D single-channel image."
        )

    # Create a boolean mask of foreground pixels (value == 255).
    # This mask is True where the digit is present and False elsewhere.
    foreground_mask = cropped_binary == 255

    # Horizontal projection: sum foreground pixels along each row.
    # axis=1 collapses columns, leaving one count per row.
    # Result length equals the image height.
    horizontal_projection = np.sum(foreground_mask, axis=1)

    # Vertical projection: sum foreground pixels along each column.
    # axis=0 collapses rows, leaving one count per column.
    # Result length equals the image width.
    vertical_projection = np.sum(foreground_mask, axis=0)

    # Maximum values: peak counts for quick shape comparison.
    maximum_horizontal_projection = int(np.max(horizontal_projection))
    maximum_vertical_projection = int(np.max(vertical_projection))

    features = {
        "horizontal_projection": horizontal_projection,
        "vertical_projection": vertical_projection,
        "maximum_horizontal_projection": maximum_horizontal_projection,
        "maximum_vertical_projection": maximum_vertical_projection,
    }

    return features


def extract_contour_features(cropped_binary):
    """
    Extract contour-based features from a cropped binary digit image.

    Uses OpenCV findContours() with RETR_EXTERNAL and CHAIN_APPROX_SIMPLE to
    detect the outer boundary of the foreground digit. Extracts the number of
    external contours, the largest contour's area and perimeter, and its
    bounding rectangle.

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Cropped binary image (0 = background, 255 = foreground).

    Returns
    -------
    dict
        Structured dictionary with keys:
            - "contour_count" : int
                Number of external contours detected.
            - "largest_contour_area" : float
                Area (in pixels) of the largest contour.
            - "largest_contour_perimeter" : float
                Perimeter (in pixels) of the largest contour.
            - "largest_contour_bbox" : tuple of int
                (x, y, w, h) bounding rectangle of the largest contour.

    Raises
    ------
    TypeError
        If the input is not a NumPy array.
    ValueError
        If the input image is empty, not 2D, or contains no contours.
    """

    # Validate input type.
    if not isinstance(cropped_binary, np.ndarray):
        raise TypeError(
            "cropped_binary must be a NumPy array."
        )

    # Validate that the image is not empty.
    if cropped_binary.size == 0:
        raise ValueError(
            "cropped_binary is empty: cannot extract contour features "
            "from an empty image."
        )

    # Ensure the image is 2D (single-channel binary).
    if cropped_binary.ndim != 2:
        raise ValueError(
            "cropped_binary must be a 2D single-channel image."
        )

    # Detect external contours only (ignores internal holes).
    # RETR_EXTERNAL retrieves only the extreme outer contours.
    # CHAIN_APPROX_SIMPLE compresses contour points to save memory.
    contours, _ = cv2.findContours(
        cropped_binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Handle case where no foreground contours are detected.
    if not contours:
        raise ValueError(
            "No contours detected: no foreground boundary found in "
            "the cropped binary image."
        )

    # Number of external contours.
    contour_count = len(contours)

    # Largest contour by area (the main digit boundary).
    largest_contour = max(contours, key=cv2.contourArea)

    # Area of the largest contour in pixels.
    largest_contour_area = float(cv2.contourArea(largest_contour))

    # Perimeter of the largest contour. True = closed contour.
    largest_contour_perimeter = float(
        cv2.arcLength(largest_contour, closed=True)
    )

    # Axis-aligned bounding rectangle of the largest contour.
    x, y, w, h = cv2.boundingRect(largest_contour)
    largest_contour_bbox = (int(x), int(y), int(w), int(h))

    features = {
        "contour_count": contour_count,
        "largest_contour_area": largest_contour_area,
        "largest_contour_perimeter": largest_contour_perimeter,
        "largest_contour_bbox": largest_contour_bbox,
    }

    return features


def extract_hole_features(cropped_binary, min_hole_area=100):
    """
    Extract enclosed hole features from a cropped binary digit image.

    Uses OpenCV findContours() with RETR_TREE to retrieve the full contour
    hierarchy. Each contour has a parent index; the nesting depth of a
    contour is the number of parent links to reach the top level (depth 0).

    Depth interpretation:
      - Depth 0: outer foreground contour (digit boundary)
      - Depth 1: enclosed background hole (inside a foreground contour)
      - Depth 2: foreground island inside a hole
      - Depth 3: hole inside that foreground island
      - etc.

    Only contours at odd depths (1, 3, 5, ...) represent enclosed holes.
    Even-depth contours are foreground boundaries and are not counted.

    Detected holes smaller than min_hole_area are filtered out as likely
    preprocessing artifacts rather than meaningful enclosed regions.

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Cropped binary image (0 = background, 255 = foreground).
    min_hole_area : int
        Minimum area (in pixels) for a detected hole to be kept.
        Default is 100. Set to 0 to disable filtering.

    Returns
    -------
    dict
        Structured dictionary with keys:
            - "num_enclosed_holes" : int
                Number of holes remaining after filtering (area >= min_hole_area).
            - "hole_areas" : list of float
                Area (in pixels) of each hole remaining after filtering.
            - "hole_centers" : list of tuple of int
                (cx, cy) pixel coordinates of each hole centroid.
            - "normalized_hole_positions" : list of tuple of float
                (nx, ny) normalized position of each hole centroid in [0, 1]
                relative to image width and height.
            - "total_detected_holes" : int
                Total number of odd-depth contours detected before filtering.
            - "filtered_out_holes" : int
                Number of holes removed because their area was below
                min_hole_area.

    Raises
    ------
    TypeError
        If the input is not a NumPy array.
    ValueError
        If the input image is empty, not 2D, or min_hole_area is negative.
    """

    # Validate input type.
    if not isinstance(cropped_binary, np.ndarray):
        raise TypeError(
            "cropped_binary must be a NumPy array."
        )

    # Validate that min_hole_area is non-negative.
    if min_hole_area < 0:
        raise ValueError(
            "min_hole_area must be non-negative."
        )

    # Validate that the image is not empty.
    if cropped_binary.size == 0:
        raise ValueError(
            "cropped_binary is empty: cannot extract hole features from "
            "an empty image."
        )

    # Ensure the image is 2D (single-channel binary).
    if cropped_binary.ndim != 2:
        raise ValueError(
            "cropped_binary must be a 2D single-channel image."
        )

    # Retrieve all contours with full hierarchy using RETR_TREE.
    # RETR_TREE reconstructs the full hierarchy of nested contours, where
    # each contour knows its parent. This lets us compute nesting depth and
    # distinguish enclosed holes (odd depth) from foreground boundaries
    # (even depth).
    # Note: cv2.findContours modifies the input image in-place, so we work
    # on a copy to preserve the original for subsequent calls.
    contours, hierarchy = cv2.findContours(
        cropped_binary.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # No contours means no foreground and therefore no holes.
    if not contours:
        return {
            "num_enclosed_holes": 0,
            "hole_areas": [],
            "total_detected_holes": 0,
            "filtered_out_holes": 0,
        }

    # hierarchy shape is (1, N, 4); flatten to (N, 4) for easier indexing.
    # Each row is [Next, Previous, First_Child, Parent].
    hierarchy = hierarchy[0]

    num_contours = len(contours)

    # Compute nesting depth for each contour by walking up the parent chain.
    # Depth 0 = top-level (parent == -1). Each parent link increases depth by 1.
    depths = [0] * num_contours
    for i in range(num_contours):
        depth = 0
        parent = hierarchy[i][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        depths[i] = depth

    # Collect areas and centroids of contours at odd depth (genuine enclosed holes).
    # Even-depth contours are foreground boundaries, not holes.
    all_hole_areas = []
    all_hole_centers = []

    img_height, img_width = cropped_binary.shape

    for i in range(num_contours):
        depth = depths[i]
        if depth % 2 == 1:
            area = float(cv2.contourArea(contours[i]))
            # Compute centroid using image moments.
            moments = cv2.moments(contours[i])
            if moments["m00"] != 0:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx = 0.0
                cy = 0.0
            all_hole_areas.append(area)
            all_hole_centers.append((cx, cy))

    total_detected_holes = len(all_hole_areas)

    # Filter out holes smaller than min_hole_area. These are typically
    # preprocessing artifacts (noise specks, anti-aliasing edge effects)
    # rather than genuine enclosed digit regions.
    filtered_hole_areas = []
    filtered_hole_centers = []

    for area, center in zip(all_hole_areas, all_hole_centers):
        if area >= min_hole_area:
            filtered_hole_areas.append(area)
            filtered_hole_centers.append(center)

    filtered_out_holes = total_detected_holes - len(filtered_hole_areas)
    num_enclosed_holes = len(filtered_hole_areas)

    # Normalize hole positions to [0, 1] relative to image dimensions.
    normalized_hole_positions = [
        (cx / img_width, cy / img_height) for (cx, cy) in filtered_hole_centers
    ]

    features = {
        "num_enclosed_holes": num_enclosed_holes,
        "hole_areas": filtered_hole_areas,
        "hole_centers": filtered_hole_centers,
        "normalized_hole_positions": normalized_hole_positions,
        "total_detected_holes": total_detected_holes,
        "filtered_out_holes": filtered_out_holes,
    }

    return features


def extract_all_features(cropped_binary, crop_info=None, min_hole_area=100):
    """
    Combine all feature extraction functions into one structured result.

    Calls extract_basic_features(), extract_projection_features(),
    extract_contour_features(), and extract_hole_features() on the same
    cropped binary image and merges their outputs into a single dictionary.

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Cropped binary image (0 = background, 255 = foreground).
    crop_info : dict or None
        Optional dictionary returned by crop_digit(). If provided, its "bbox"
        is passed to extract_basic_features() for bounding box reuse.
    min_hole_area : int
        Minimum area (in pixels) for a detected hole to be kept.
        Passed to extract_hole_features(). Default is 100.

    Returns
    -------
    dict
        Structured dictionary with grouped feature keys:
            - "basic" : dict from extract_basic_features()
            - "projection" : dict from extract_projection_features()
            - "contour" : dict from extract_contour_features()
            - "hole" : dict from extract_hole_features()

    Raises
    ------
    TypeError
        If cropped_binary is not a NumPy array.
    ValueError
        If cropped_binary is empty, not 2D, or min_hole_area is negative.
    """

    # Validate input type.
    if not isinstance(cropped_binary, np.ndarray):
        raise TypeError(
            "cropped_binary must be a NumPy array."
        )

    # Validate that the image is not empty.
    if cropped_binary.size == 0:
        raise ValueError(
            "cropped_binary is empty: cannot extract features from "
            "an empty image."
        )

    # Ensure the image is 2D (single-channel binary).
    if cropped_binary.ndim != 2:
        raise ValueError(
            "cropped_binary must be a 2D single-channel image."
        )

    # Validate min_hole_area before delegating.
    if min_hole_area < 0:
        raise ValueError(
            "min_hole_area must be non-negative."
        )

    # Delegate to each individual feature extraction function.
    basic_features = extract_basic_features(
        cropped_binary, crop_info=crop_info
    )
    projection_features = extract_projection_features(cropped_binary)
    contour_features = extract_contour_features(cropped_binary)
    hole_features = extract_hole_features(
        cropped_binary, min_hole_area=min_hole_area
    )

    # Merge results into a clearly organized dictionary.
    all_features = {
        "basic": basic_features,
        "projection": projection_features,
        "contour": contour_features,
        "hole": hole_features,
    }

    return all_features


def extract_normalized_features(cropped_binary, crop_info=None, min_hole_area=100):
    """
    Extract resolution-independent normalized features from a cropped
    binary digit image.

    Reuses the existing feature extraction functions and normalizes their
    outputs so values remain comparable across images of different
    dimensions.

    Normalization formulas:
      - aspect_ratio            : width / height (already dimensionless)
      - foreground_density      : foreground_pixels / total_pixels
      - norm_h_projection[i]    : hp[i] / max(hp)        -> range [0, 1]
      - norm_v_projection[j]    : vp[j] / max(vp)        -> range [0, 1]
      - norm_contour_area       : contour_area / (w * h) -> range [0, 1]
      - norm_hole_areas[k]      : hole_area[k] / (w * h) -> range [0, 1]
      - norm_hole_positions[k]  : (nx, ny) in [0, 1] relative to image dims
      - hole_count              : number of enclosed holes (count)

    Parameters
    ----------
    cropped_binary : numpy.ndarray
        Cropped binary image (0 = background, 255 = foreground).
    crop_info : dict or None
        Optional dictionary returned by crop_digit(). If provided, its "bbox"
        is passed to extract_basic_features() for bounding box reuse.
    min_hole_area : int
        Minimum area for hole detection. Passed to extract_hole_features().
        Default is 100.

    Returns
    -------
    dict
        Structured dictionary with keys:
            - "aspect_ratio" : float
                Width divided by height of the digit bounding box.
            - "foreground_density" : float
                Fraction of pixels that are foreground (range [0, 1]).
            - "normalized_horizontal_projection" : numpy.ndarray
                Horizontal projection profile divided by its maximum value.
                Length equals image height; values in range [0, 1].
            - "normalized_vertical_projection" : numpy.ndarray
                Vertical projection profile divided by its maximum value.
                Length equals image width; values in range [0, 1].
            - "normalized_contour_area" : float
                Largest contour area divided by bounding box area
                (width * height). Range [0, 1].
            - "normalized_hole_areas" : list of float
                Each enclosed hole area divided by bounding box area
                (width * height). Range [0, 1] per hole.
            - "normalized_hole_positions" : list of tuple of float
                Each hole centroid position as (nx, ny) in [0, 1] relative
                to image width and height.
            - "hole_count" : int
                Number of enclosed holes after min_hole_area filtering.

    Raises
    ------
    TypeError
        If cropped_binary is not a NumPy array.
    ValueError
        If cropped_binary is empty, not 2D, or min_hole_area is negative.
    """

    # Validate input type.
    if not isinstance(cropped_binary, np.ndarray):
        raise TypeError(
            "cropped_binary must be a NumPy array."
        )

    # Validate that the image is not empty.
    if cropped_binary.size == 0:
        raise ValueError(
            "cropped_binary is empty: cannot extract normalized features "
            "from an empty image."
        )

    # Ensure the image is 2D (single-channel binary).
    if cropped_binary.ndim != 2:
        raise ValueError(
            "cropped_binary must be a 2D single-channel image."
        )

    # Validate min_hole_area before delegating.
    if min_hole_area < 0:
        raise ValueError(
            "min_hole_area must be non-negative."
        )

    # Delegate to existing feature extraction functions.
    basic_features = extract_basic_features(
        cropped_binary, crop_info=crop_info
    )
    projection_features = extract_projection_features(cropped_binary)
    contour_features = extract_contour_features(cropped_binary)
    hole_features = extract_hole_features(
        cropped_binary, min_hole_area=min_hole_area
    )

    # Extract raw values from grouped feature dictionaries.
    width = basic_features["width"]
    height = basic_features["height"]
    bbox_area = float(width * height)

    if bbox_area == 0:
        raise ValueError(
            "Bounding box area is zero: cannot normalize features."
        )

    # Aspect ratio: width / height (already dimensionless).
    aspect_ratio = float(basic_features["aspect_ratio"])

    # Foreground density: fraction of pixels that are foreground.
    foreground_density = float(basic_features["foreground_pixel_density"])

    # Normalize projection profiles by their maximum value.
    # This makes profiles comparable across images of different scales:
    # the peak row/column always equals 1.0 regardless of absolute size.
    raw_h_projection = projection_features["horizontal_projection"]
    raw_v_projection = projection_features["vertical_projection"]

    max_h_projection = float(projection_features[
        "maximum_horizontal_projection"])
    max_v_projection = float(projection_features[
        "maximum_vertical_projection"])

    if max_h_projection > 0:
        normalized_h_projection = (
            raw_h_projection.astype(np.float64) / max_h_projection
        )
    else:
        normalized_h_projection = raw_h_projection.astype(np.float64)

    if max_v_projection > 0:
        normalized_v_projection = (
            raw_v_projection.astype(np.float64) / max_v_projection
        )
    else:
        normalized_v_projection = raw_v_projection.astype(np.float64)

    # Smooth projection profiles to reduce texture/noise sensitivity.
    # A simple 3-point moving average makes the projection-based features
    # more robust to anti-aliasing, texture, and stroke-width variations.
    if len(normalized_h_projection) >= 3:
        kernel = np.ones(3) / 3.0
        normalized_h_projection = np.convolve(
            normalized_h_projection, kernel, mode='same'
        )
    if len(normalized_v_projection) >= 3:
        kernel = np.ones(3) / 3.0
        normalized_v_projection = np.convolve(
            normalized_v_projection, kernel, mode='same'
        )

    # Normalized contour area: fraction of bounding box occupied by the
    # largest contour.
    largest_contour_area = float(contour_features["largest_contour_area"])
    normalized_contour_area = largest_contour_area / bbox_area

    # Normalized hole areas: each hole area as a fraction of bounding box.
    raw_hole_areas = hole_features["hole_areas"]
    normalized_hole_areas = [
        float(area) / bbox_area for area in raw_hole_areas
    ]

    # Normalized hole positions from hole detection.
    normalized_hole_positions = hole_features[
        "normalized_hole_positions"
    ]

    # Hole count (already dimensionless).
    hole_count = int(hole_features["num_enclosed_holes"])

    features = {
        "aspect_ratio": aspect_ratio,
        "foreground_density": foreground_density,
        "normalized_horizontal_projection": normalized_h_projection,
        "normalized_vertical_projection": normalized_v_projection,
        "normalized_contour_area": float(normalized_contour_area),
        "normalized_hole_areas": normalized_hole_areas,
        "normalized_hole_positions": normalized_hole_positions,
        "hole_count": hole_count,
    }

    return features
