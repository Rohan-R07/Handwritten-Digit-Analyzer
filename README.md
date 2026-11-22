# Rule-Based Handwritten Digit Recognition

A rule-based handwritten digit recognition system that uses traditional image processing techniques — no machine learning, no pretrained models, no training datasets. The pipeline preprocesses handwritten digit images, extracts geometric features, and classifies digits using rule-based logic and template matching.

## Project Overview

This project implements a complete handwritten digit recognition pipeline using only OpenCV and NumPy. The system processes handwritten digit images through:

1. **Preprocessing** — Grayscale conversion, Gaussian blur, Otsu thresholding, morphological closing, connected-component noise removal, and digit cropping/padding
2. **Feature Extraction** — Geometric features (aspect ratio, density), projection profiles, contour analysis, and hole detection
3. **Rule-Based Classification** — Rule-based scoring for digits 0–9 using geometric features
4. **Template Matching** — Pixel-based and shape-based comparison against reference digit templates
5. **Hybrid Recognition** — Combined template matching + rule-based classification

The system is designed to handle various handwriting styles including varying stroke thickness, different orientations, different image scales, and different colors (dark on light backgrounds).

## Project Structure

```
Ai based handwritten digit recognition/
├── app.py                      # Streamlit testing interface
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── scripts/
│   └── generate_templates.py   # Template generation utility
├── src/
│   ├── __pycache__/
│   ├── digit_classifier.py     # Placeholder for future ML/classification
│   ├── digit_recognizer.py     # Hybrid recognition engine (template + features)
│   ├── feature_extraction.py   # Geometric feature extraction
│   ├── main.py                 # Basic CLI entry point
│   ├── preprocessing.py        # Image preprocessing pipeline
│   ├── rule_classifier.py      # Rule-based digit classification
│   ├── template_manager.py     # Template loading and validation
│   └── template_matcher.py     # Template matching engine
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── index.css
│       ├── components/
│       │   ├── App.tsx
│       │   ├── CroppedDigit.tsx
│       │   ├── DigitRecognition.tsx
│       │   ├── FeatureDetails.tsx
│       │   ├── ImagePreview.tsx
│       │   ├── Navbar.tsx
│       │   ├── ProcessingState.tsx
│       │   ├── RecentRecognitions.tsx
│       │   ├── RecognitionResult.tsx
│       │   └── UploadDropzone.tsx
│       ├── hooks/
│       │   └── useDigitRecognition.ts
│       ├── services/
│       │   └── recognitionApi.ts
│       └── types/
│           └── recognition.ts
├── templates/
│   ├── 0.png   through   9.png   # Reference digit templates (28x28 binary)
└── test_images/
    ├── 0image.png
    ├── 6imagin.png
    ├── 9digit.png
    ├── image3.png
    ├── imagenormal3.png
    ├── testimg1.png
    └── testimg4.png
```

## Prerequisites

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Streamlit (for the web interface)
- FastAPI + Uvicorn (for the REST API backend)

## Installation

```bash
# Clone or navigate to the project directory
cd "Ai based handwritten digit recognition"

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Streamlit Interface

```bash
streamlit run app.py
```

This launches a web interface where you can:
- Upload handwritten digit images (PNG, JPG, JPEG)
- View preprocessing stages (grayscale, blur, binary, cleaned)
- View the cropped and normalized digit
- View extracted features (basic, projection, contour, hole features)
- View digit recognition results

### Option 2: Run the FastAPI Backend (for React frontend)

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

This starts a REST API server at `http://localhost:8000` that accepts image uploads and returns recognition results as JSON.

### Option 3: Run from Command Line

```bash
cd src
python main.py
```

This runs the preprocessing pipeline on `test_images/testimg1.png` and displays the intermediate images.

### Using as a Library

```python
from src.preprocessing import preprocess_image, crop_digit, normalize_digit_image
from src.digit_recognizer import recognize_digit

# Preprocess an image
image, gray, blurred, binary, cleaned = preprocess_image("test_images/testimg1.png")

# Crop the digit
cropped, crop_info = crop_digit(cleaned, padding=10)

# Normalize to 28x28
normalized = normalize_digit_image(cropped)

# Recognize the digit
result = recognize_digit(normalized, crop_info=crop_info)
print(f"Predicted digit: {result['predicted_digit']}")
print(f"Similarity score: {result['similarity_score']}")
print(f"Explanation: {result['explanation']}")
```

---

## Pipeline Overview

### 1. Preprocessing (`src/preprocessing.py`)

The preprocessing pipeline transforms a raw input image into a clean, cropped, normalized 28×28 binary image:

1. **Load Image** — Reads the input image file
2. **Grayscale Conversion** — Converts to grayscale using channel-aware conversion for optimal foreground/background separation (handles colored images like blue or black handwriting)
3. **Gaussian Blur** — Applies 5×5 Gaussian blur for noise reduction
4. **Otsu Thresholding** — Converts to binary using Otsu's method with binary inversion (foreground = white, background = black)
5. **Morphological Closing** — Fills small gaps in digit strokes
6. **Connected-Component Noise Removal** — Removes isolated noise blobs while preserving the main digit
7. **Digit Cropping** — Detects and crops to the digit bounding box using connected-component analysis
8. **Digit Normalization** — Resizes the digit proportionally to fit within 20×20, centers it on a 28×28 black canvas while preserving aspect ratio

### 2. Feature Extraction (`src/feature_extraction.py`)

Extracts geometric features from the cropped binary digit:

- **Basic Features**: Bounding box, width, height, aspect ratio, foreground pixel count, foreground pixel density
- **Projection Features**: Horizontal and vertical projection profiles (normalized)
- **Contour Features**: Contour count, largest contour area, perimeter, bounding rectangle
- **Hole Features**: Enclosed hole count, hole areas, hole centers, hole positions (using contour hierarchy with RETR_TREE)

### 3. Rule-Based Classification (`src/rule_classifier.py`)

Classifies digits using rule-based scoring:

| Digit | Key Features |
|-------|-------------|
| 0 | 1 enclosed hole, centered, dense, square-ish |
| 1 | No holes, tall aspect ratio, sparse density |
| 2 | No holes, top-heavy horizontal projection, curve-diagonal-base shape |
| 3 | No holes, right-heavy vertical projection, two curved lobes |
| 4 | No holes or 1 hole, right-heavy vertical projection |
| 5 | No holes, bottom-heavy horizontal projection |
| 6 | 1 hole at bottom, dense |
| 7 | No holes, tall/square-ish, sparse, top-heavy horizontal projection |
| 8 | 2 vertically-stacked enclosed holes, high density, square-ish |
| 9 | 1 hole at top, dense |

### 4. Template Matching (`src/template_matcher.py`)

Compares the normalized input digit against reference templates (0–9) using:

- **Pixel Similarity**: Pixel-by-pixel comparison with multi-shift tolerance
- **Shape Similarity**: OpenCV `findContours` + `cv2.matchShapes` (Hu moment invariants) + distance transform correlation
- Combined score: 40% pixel similarity + 60% shape similarity

### 5. Hybrid Recognition (`src/digit_recognizer.py`)

Combines template matching and rule-based classification:

- Template matching provides the primary signal (pixel + shape similarity)
- Feature extraction provides supporting evidence
- Connected-component analysis validates digit-like shapes
- Returns prediction with confidence score and alternative candidates

---

## API Reference

### POST /api/recognize

Upload a handwritten digit image for recognition.

**Request:**
```
POST /api/recognize
Content-Type: multipart/form-data
Body: file=<image file>
```

**Response:**
```json
{
  "status": "ok",
  "predictedDigit": 3,
  "message": "Recognized with strong template match. Template similarity for digit 3: 0.82. Combined score: 0.85.",
  "candidates": [
    {"digit": 3, "score": 0.85},
    {"digit": 8, "score": 0.45},
    {"digit": 5, "score": 0.30}
  ],
  "processingImages": {},
  "details": {
    "template_scores": {"0": 0.1, "1": 0.05, "2": 0.15, "3": 0.82, "4": 0.1, "5": 0.2, "6": 0.05, "7": 0.02, "8": 0.45, "9": 0.3},
    "shape_scores": {"0": 0.1, "1": 0.05, "2": 0.15, "3": 0.8, "4": 0.1, "5": 0.15, "6": 0.02, "7": 0.01, "8": 0.4, "9": 0.25},
    "feature_scores": {"0": 0.1, "1": 0.05, "2": 0.15, "3": 0.85, "4": 0.1, "5": 0.25, "6": 0.05, "7": 0.02, "8": 0.45, "9": 0.3}
  },
  "raw": { ...full internal result... }
}
```

### GET /health

Health check endpoint.

Returns: `{"status": "ok"}`

---

## Test Images

The `test_images/` directory contains sample images for testing:

| Image | Description |
|-------|-------------|
| `testimg1.png` | Sample handwritten digit image |
| `testimg4.png` | Sample handwritten digit image |
| `testimg3.png` | Sample handwritten digit image |
| `imagenormal3.png` | Normal handwritten digit |
| `image3.png` | Charcoal/brushstroke digit |
| `6imagin.png` | Pencil-written digit |
| `0image.png` | Rule-based digit |

Reference templates for digits 0–9 are stored in `templates/`.

## Limitations

- The system is rule-based and does not use machine learning, so accuracy may vary with unusual handwriting styles
- Single-digit oriented — does not separate touching digits or handle multi-digit images
- Template matching depends on the quality and variety of reference templates
- Highly stylized or artistic handwriting may not match well with rule-based features
- No training dataset — templates are machine-generated geometric shapes

## License

This project is for educational purposes. No machine learning models or training datasets are used.
