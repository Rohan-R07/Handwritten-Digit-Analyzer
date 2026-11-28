
import cv2
import numpy as np
import os
import sys
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    convert_to_grayscale,
    remove_noise,
    try_preprocessing_variants,
    clean_binary,
    remove_small_components,
    crop_digit,
    normalize_digit_image,
)
from digit_recognizer import recognize_digit
from feature_extraction import extract_all_features, extract_normalized_features

app = FastAPI(title="Handwritten Digit Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_data_url(image: np.ndarray) -> str:
    """Convert a NumPy image array to a base64 data URL."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        _, buffer = cv2.imencode(".png", image)
    else:
        _, buffer = cv2.imencode(".png", image)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/recognize")
async def recognize_digit_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "status": "invalid",
            "predictedDigit": None,
            "message": "Could not decode image. Please upload a valid PNG, JPG, or JPEG file.",
            "candidates": [],
            "processingImages": {},
            "details": {},
            "raw": {},
        }

    try:
        # Step 1: Grayscale conversion
        gray = convert_to_grayscale(image)

        # Step 2: Gaussian blur
        blurred = remove_noise(gray)

        # Step 3: Select the best existing preprocessing variant
        _, binary = try_preprocessing_variants(image)

        # Step 4: Morphological closing
        cleaned = clean_binary(binary)

        # Step 5: Remove small noise components
        cleaned = remove_small_components(cleaned, min_area=50)

        # Step 6: Crop digit bounding box
        cropped, crop_info = crop_digit(cleaned, padding=10)

        # Step 7: Normalize to 28x28 canvas
        normalized = normalize_digit_image(cropped)

        # Step 8: Run digit recognition
        result = recognize_digit(normalized, crop_info=crop_info, min_hole_area=20)

        # Build processing images dictionary for frontend visualization
        processing_images = {
            "original": image_to_data_url(image),
            "grayscale": image_to_data_url(gray),
            "blurred": image_to_data_url(blurred),
            "thresholded": image_to_data_url(binary),
            "cleaned": image_to_data_url(cleaned),
            "cropped": image_to_data_url(cropped),
            "normalized": image_to_data_url(normalized),
        }

        # Build details dictionary with feature information
        details = {
            "template_scores": result.get("template_scores", {}),
            "shape_scores": result.get("shape_scores", {}),
            "feature_scores": result.get("feature_scores", {}),
        }

        return {
            "status": "success",
            "predictedDigit": result.get("predicted_digit"),
            "message": result.get("explanation", ""),
            "candidates": [
                {"digit": d, "score": s}
                for d, s in result.get("ranked_candidates", [])
            ],
            "processingImages": processing_images,
            "details": details,
            "raw": result,
        }

    except ValueError as e:
        return {
            "status": "invalid",
            "predictedDigit": None,
            "message": str(e),
            "candidates": [],
            "processingImages": {},
            "details": {},
            "raw": {},
        }
    except Exception as e:
        return {
            "status": "invalid",
            "predictedDigit": None,
            "message": str(e),
            "candidates": [],
            "processingImages": {},
            "details": {},
            "raw": {},
        }
