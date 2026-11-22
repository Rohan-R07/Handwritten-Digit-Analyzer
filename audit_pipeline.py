"""
Comprehensive audit script: trace all images through the digit recognition pipeline.
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, 'src')

from preprocessing import preprocess_image, crop_digit, normalize_digit_image
from digit_recognizer import recognize_digit

# Expected digits for each image
TEST_CASES = [
    ("test_images/image3.png", 3),
    ("test_images/imagenormal3.png", 3),
    ("test_images/0image.png", 0),
    ("test_images/testimg1.png", None),  # unknown expected
    ("test_images/testimg4.png", None),  # unknown expected
]

print("=" * 90)
print("PIPELINE AUDIT FOR ALL SUPPLIED IMAGES")
print("=" * 90)

for img_path, expected in TEST_CASES:
    if not os.path.exists(img_path):
        print(f"\n{'='*90}\nIMAGE: {img_path} - FILE NOT FOUND\n{'='*90}")
        continue

    print(f"\n{'='*90}")
    print(f"IMAGE: {img_path} (expected: {expected})")
    print(f"{'='*90}")

    try:
        # Step 1: Load and preprocess
        image, gray, blurred, binary, cleaned = preprocess_image(img_path)
        print(f"\n[PREPROCESS] Original shape: {image.shape}")
        print(f"[PREPROCESS] Binary foreground pixels: {np.count_nonzero(binary == 255)}")
        print(f"[PREPROCESS] Cleaned foreground pixels: {np.count_nonzero(cleaned == 255)}")

        # Step 2: Crop
        cropped, info = crop_digit(cleaned, padding=10)
        print(f"\n[CROP] Cropped shape: {cropped.shape}")
        print(f"[CROP] Bounding box: {info['bbox']}")
        print(f"[CROP] Touches boundary: {info['touches_boundary']}")
        print(f"[CROP] Component score: {info['component_score']:.3f}")
        print(f"[CROP] Num candidates: {info['num_candidates']}")
        print(f"[CROP] Cropped foreground pixels: {np.count_nonzero(cropped == 255)}")

        # Step 3: Normalize
        normalized = normalize_digit_image(cropped)
        print(f"\n[NORMALIZE] Normalized shape: {normalized.shape}")
        print(f"[NORMALIZE] Normalized foreground pixels: {np.count_nonzero(normalized == 255)}")

        # Step 4: Recognition
        result = recognize_digit(normalized, crop_info=info, min_hole_area=20)
        print(f"\n[RECOGNIZE] Status: {result['status']}")
        print(f"[RECOGNIZE] Predicted digit: {result['predicted_digit']}")
        print(f"[RECOGNIZE] Similarity score: {result['similarity_score']}")
        print(f"[RECOGNIZE] Expected: {expected}")
        print(f"[RECOGNIZE] Correct: {result['predicted_digit'] == expected}")
        print(f"[RECOGNIZE] Explanation: {result['explanation'][:200]}")

        # Template scores
        template_scores = result.get("template_scores", {})
        sorted_tmpl = sorted(template_scores.items(), key=lambda x: -x[1])
        print(f"\n[TEMPLATE] Top 5 template scores:")
        for d, s in sorted_tmpl[:5]:
            print(f"  Digit {d}: {s:.4f}")

        # Feature scores
        feature_scores = result.get("feature_scores", {})
        if feature_scores:
            sorted_feat = sorted(feature_scores.items(), key=lambda x: -x[1])
            print(f"\n[FEATURE] Top 5 feature scores:")
            for d, s in sorted_feat[:5]:
                print(f"  Digit {d}: {s:.4f}")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()

print(f"\n{'='*90}")
print("AUDIT COMPLETE")
print(f"{'='*90}")
