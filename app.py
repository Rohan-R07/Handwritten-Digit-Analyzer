
import cv2
import numpy as np
import streamlit as st
import tempfile
import os

from src.preprocessing import (
    load_image,
    convert_to_grayscale,
    remove_noise,
    apply_threshold,
    clean_binary,
    remove_small_components,
    crop_digit,
    normalize_digit_image,
    preprocess_image,
)
from src.feature_extraction import extract_all_features
from src.digit_recognizer import recognize_digit


st.set_page_config(
    page_title="Handwritten Digit Recognition - Rule-Based Pipeline",
    layout="wide",
)

st.title("Handwritten Digit Recognition - Rule-Based Pipeline")

st.markdown(
    """
    Upload a handwritten digit image (PNG, JPG, JPEG) to test the rule-based
    image preprocessing and digit recognition pipeline.

    The pipeline performs:
    1. Grayscale conversion
    2. Gaussian blur (noise reduction)
    3. Otsu thresholding with binary inversion
    4. Morphological closing (gap filling)
    5. Connected-component noise removal
    6. Digit cropping with padding
    7. Digit normalization to 28x28 canvas
    8. Feature extraction (basic, projection, contour, hole features)
    9. Template matching + rule-based classification

    No machine learning is used. This is a purely rule-based pipeline.
    """
)


def load_uploaded_image(uploaded_file):
    """
    Save an uploaded Streamlit file to a temporary path and load it with OpenCV.

    Returns
    -------
    tuple of (numpy.ndarray, str)
        The loaded BGR image and the temporary file path.
    """

    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    image = load_image(tmp_path)
    return image, tmp_path


def display_feature_group(title, feature_dict):
    """
    Display a feature group dictionary in a readable format.

    Numerical scalars are shown as key-value pairs. NumPy arrays (e.g.,
    projection profiles) are summarized with length, min, max, mean, and
    a small preview instead of dumping the entire array.
    """

    st.subheader(title)

    for key, value in feature_dict.items():
        if isinstance(value, np.ndarray):
            st.write(
                "**{}**: length={}, min={}, max={}, mean={:.2f}".format(
                    key,
                    len(value),
                    int(np.min(value)),
                    int(np.max(value)),
                    float(np.mean(value)),
                )
            )
            with st.expander("Preview first 20 values"):
                st.write(value[:20].tolist())
        elif isinstance(value, (int, float, np.integer, np.floating)):
            st.write("**{}**: {}".format(key, value))
        elif isinstance(value, (list, tuple)):
            st.write("**{}**: {}".format(key, value))
        else:
            st.write("**{}**: {}".format(key, value))


uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG, JPG, or JPEG image of a handwritten digit.",
)

if uploaded_file is not None:
    try:
        image, tmp_path = load_uploaded_image(uploaded_file)

        st.success("Image loaded successfully: {}".format(uploaded_file.name))

        # Initialize variables
        normalized = None
        cropped = None
        crop_info = None

        # Display original image
        st.subheader("1. Original Image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Original uploaded image", use_column_width=True)

        # Run preprocessing pipeline
        st.subheader("2. Preprocessing Pipeline")
        with st.spinner("Running preprocessing..."):
            original, grayscale, blurred, binary, cleaned = preprocess_image(
                tmp_path
            )

        # Display preprocessing stages in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Grayscale**")
            st.image(grayscale, caption="Grayscale", use_column_width=True)

            st.write("**Binary (Otsu + Inverted)**")
            st.image(binary, caption="Binary threshold", use_column_width=True)

        with col2:
            st.write("**Blurred (Gaussian 5x5)**")
            st.image(blurred, caption="Blurred", use_column_width=True)

            st.write("**Cleaned Binary (Morphological Close)**")
            st.image(cleaned, caption="Cleaned binary", use_column_width=True)

        with col3:
            st.write("**Cropped Digit**")
            try:
                cropped, crop_info = crop_digit(cleaned, padding=10)

                # Normalize the cropped digit to 28x28
                try:
                    normalized = normalize_digit_image(cropped)
                    norm_ok = True
                    norm_error = None
                except Exception as norm_error:
                    normalized = None
                    norm_ok = False

                # Display cropped and normalized digits side by side
                norm_col1, norm_col2 = st.columns(2)

                with norm_col1:
                    st.image(
                        cropped,
                        caption="Cropped digit ({}x{})".format(
                            cropped.shape[1], cropped.shape[0]
                        ),
                        use_column_width=True,
                    )
                    st.write("Bounding box: {}".format(crop_info["bbox"]))

                with norm_col2:
                    if norm_ok and normalized is not None:
                        st.image(
                            normalized,
                            caption="Normalized Digit (28x28)",
                            use_column_width=True,
                        )
                        st.write("Output dimensions: 28x28")
                        st.write("Digit visible: {}".format(
                            int(np.count_nonzero(normalized == 255)) > 0
                        ))
                    else:
                        st.write("**Normalized Digit (28x28)**")
                        st.error(
                            "Normalization failed: {}".format(norm_error)
                        )

                if crop_info["touches_boundary"]:
                    st.warning(
                        "Warning: digit touches the image boundary. "
                        "Input may be clipped; missing portions are not "
                        "reconstructed."
                    )

            except ValueError as crop_error:
                st.error("Digit cropping failed: {}".format(crop_error))
                cropped = None
                crop_info = None

        # Feature extraction
        st.subheader("3. Feature Extraction")
        with st.expander("View feature extraction details", expanded=False):
            if cropped is not None:
                try:
                    with st.spinner("Extracting features..."):
                        all_features = extract_all_features(
                            cropped, crop_info=crop_info, min_hole_area=100
                        )

                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["Basic", "Projection", "Contour", "Hole"]
                    )

                    with tab1:
                        display_feature_group(
                            "Basic Geometric Features",
                            all_features["basic"],
                        )

                    with tab2:
                        display_feature_group(
                            "Projection Profile Features",
                            all_features["projection"],
                        )

                    with tab3:
                        display_feature_group(
                            "Contour Features",
                            all_features["contour"],
                        )

                    with tab4:
                        display_feature_group(
                            "Hole Features",
                            all_features["hole"],
                        )

                except Exception as feature_error:
                    st.error("Feature extraction failed: {}".format(feature_error))
            else:
                st.info(
                    "Feature extraction skipped because digit cropping failed."
                )

        # Digit recognition
        st.subheader("4. Digit Recognition")
        if normalized is not None:
            try:
                with st.spinner("Recognizing digit..."):
                    recognition = recognize_digit(
                        normalized, crop_info=crop_info, min_hole_area=20
                    )

                status = recognition["status"]

                if status == "ok":
                    # Strong recognition: show prominent prediction
                    st.success("**Status: Recognized**")

                    pred_col1, pred_col2, pred_col3 = st.columns(3)

                    with pred_col1:
                        st.metric(
                            "Predicted Digit",
                            recognition["predicted_digit"],
                        )

                    with pred_col2:
                        st.metric(
                            "Similarity Score",
                            "{:.2f}".format(recognition["similarity_score"]),
                        )

                    with pred_col3:
                        st.write("**Status:** {}".format(
                            recognition["status"].upper()
                        ))

                    st.write("**Explanation:** {}".format(
                        recognition["explanation"]
                    ))

                    # Alternative candidates
                    if recognition["alternative_candidates"]:
                        st.write("**Alternative Candidates:**")
                        alt_cols = st.columns(len(recognition["alternative_candidates"]))
                        for i, (digit, score) in enumerate(
                            recognition["alternative_candidates"]
                        ):
                            with alt_cols[i]:
                                st.metric(
                                    "Digit {}".format(digit),
                                    "{:.2f}".format(score),
                                )

                elif status == "uncertain":
                    # Moderate evidence: show best candidate with warning
                    st.warning("**Status: Uncertain**")

                    pred_col1, pred_col2 = st.columns(2)

                    with pred_col1:
                        st.metric(
                            "Best Candidate Digit",
                            recognition["predicted_digit"],
                        )

                    with pred_col2:
                        st.metric(
                            "Similarity Score",
                            "{:.2f}".format(recognition["similarity_score"]),
                        )

                    st.write("**Explanation:** {}".format(
                        recognition["explanation"]
                    ))

                    # Alternative candidates
                    if recognition["alternative_candidates"]:
                        st.write("**Alternative Candidates:**")
                        alt_cols = st.columns(len(recognition["alternative_candidates"]))
                        for i, (digit, score) in enumerate(
                            recognition["alternative_candidates"]
                        ):
                            with alt_cols[i]:
                                st.metric(
                                    "Digit {}".format(digit),
                                    "{:.2f}".format(score),
                                )

                elif status == "invalid_input":
                    # Weak evidence or blank: reject
                    st.error("**Status: Invalid Input**")
                    st.write("**Explanation:** {}".format(
                        recognition.get("error_message", "Recognition unsuccessful.")
                    ))
                    st.write("**Details:** {}".format(
                        recognition["explanation"]
                    ))

                elif status == "no_templates":
                    st.warning(
                        "Recognition skipped: {}."
                        .format(recognition.get("error_message", "No templates available.")))
                else:
                    st.error(
                        "Recognition error: {}."
                        .format(recognition.get("error_message", "Unknown error.")))

                # Template scores breakdown (always show when we have scores)
                if recognition.get("template_scores"):
                    with st.expander("View template matching details", expanded=False):
                        st.write("**Template Scores (all digits):**")
                        template_scores = recognition["template_scores"]
                        score_cols = st.columns(5)
                        for i in range(10):
                            digit = i
                            score = template_scores.get(digit, 0.0)
                            with score_cols[i % 5]:
                                st.metric(
                                    "Digit {}".format(digit),
                                    "{:.2f}".format(score),
                                )

                # Feature scores breakdown
                if recognition.get("feature_scores"):
                    with st.expander("View geometric feature scores", expanded=False):
                        st.write("**Rule-Based Feature Scores:**")
                        feature_scores = recognition["feature_scores"]
                        score_cols = st.columns(5)
                        for i in range(10):
                            digit = i
                            score = feature_scores.get(digit, 0.0)
                            with score_cols[i % 5]:
                                st.metric(
                                    "Digit {}".format(digit),
                                    "{:.2f}".format(score),
                                )

            except Exception as recognition_error:
                st.error("Digit recognition failed: {}".format(recognition_error))
        else:
            st.info(
                "Digit recognition skipped because normalization failed."
            )

        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    except FileNotFoundError as fnf_error:
        st.error("Could not load image: {}".format(fnf_error))
    except ValueError as val_error:
        st.error("Invalid image: {}".format(val_error))
    except Exception as unexpected_error:
        st.error("An unexpected error occurred: {}".format(unexpected_error))

else:
    st.info(
        "Please upload a digit image to begin. "
        "Sample images are available in the `test_images/` folder."
    )
