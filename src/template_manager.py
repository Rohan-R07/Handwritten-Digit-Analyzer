
import cv2
import numpy as np
import os


TEMPLATE_DIR = "templates"
CANVAS_SIZE = 28


def _template_path(digit):
    """Return the expected file path for a digit template."""
    return os.path.join(TEMPLATE_DIR, "{}.png".format(digit))


def load_template(digit):
    """
    Load a single digit template from the templates directory.

    Parameters
    ----------
    digit : int
        Digit value (0-9).

    Returns
    -------
    numpy.ndarray or None
        The loaded 28x28 binary template, or None if the file is missing
        or invalid.

    Raises
    ------
    ValueError
        If digit is not in range 0-9.
    """

    if not isinstance(digit, int) or digit < 0 or digit > 9:
        raise ValueError("digit must be an integer between 0 and 9.")

    path = _template_path(digit)

    if not os.path.isfile(path):
        return None

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    if not _is_valid_template(image):
        return None

    return image


def load_all_templates():
    """
    Load all available digit templates (0-9) from the templates directory.

    Returns
    -------
    dict
        Dictionary mapping digit (int) to template image (numpy.ndarray).
        Only digits with valid template files are included.
        Missing or invalid templates are omitted and reported via
        get_missing_templates().

    Raises
    ------
    None
    """

    templates = {}

    for digit in range(10):
        template = load_template(digit)
        if template is not None:
            templates[digit] = template

    return templates


def get_missing_templates():
    """
    Return a list of digit values that are missing or have invalid templates.

    Returns
    -------
    list of int
        Digit values (0-9) for which no valid template was found.
    """

    missing = []

    for digit in range(10):
        if load_template(digit) is None:
            missing.append(digit)

    return missing


def _is_valid_template(image):
    """
    Validate that an image is a 28x28 binary template with pixel values 0/255.

    Parameters
    ----------
    image : numpy.ndarray
        Image to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """

    if not isinstance(image, np.ndarray):
        return False

    if image.ndim != 2:
        return False

    if image.shape != (CANVAS_SIZE, CANVAS_SIZE):
        return False

    unique_values = set(np.unique(image).tolist())
    if not unique_values <= {0, 255}:
        return False

    return True


def validate_template_file(path):
    """
    Validate a template file path without loading it into the module.

    Parameters
    ----------
    path : str
        File path to validate.

    Returns
    -------
    dict
        Validation result with keys:
            - "valid" : bool
            - "reason" : str or None
            - "shape" : tuple or None
            - "unique_values" : list or None
    """

    result = {
        "valid": False,
        "reason": None,
        "shape": None,
        "unique_values": None,
    }

    if not os.path.isfile(path):
        result["reason"] = "File does not exist."
        return result

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        result["reason"] = "Could not read image file."
        return result

    result["shape"] = tuple(image.shape)
    result["unique_values"] = sorted(np.unique(image).tolist())

    if image.ndim != 2:
        result["reason"] = "Image must be single-channel (grayscale)."
        return result

    if image.shape != (CANVAS_SIZE, CANVAS_SIZE):
        result["reason"] = "Image dimensions must be 28x28, got {}x{}.".format(
            image.shape[0], image.shape[1]
        )
        return result

    unique_set = set(result["unique_values"])
    if not unique_set <= {0, 255}:
        result["reason"] = "Pixel values must be only 0 and 255."
        return result

    result["valid"] = True
    return result
