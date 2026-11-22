"""
Debug script for 0image.png - check hole detection and feature scoring
"""
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from preprocessing import preprocess_image, crop_digit, normalize_digit_image
from feature_extraction import extract_normalized_features, extract_hole_features
from digit_recognizer import recognize_digit
from rule_classifier import classify_features

image_path = 'test_images/0image.png'
image, gray, blurred, binary, cleaned = preprocess_image(image_path)
cropped, info = crop_digit(cleaned, padding=10)
norm = normalize_digit_image(cropped)

print("=== 0image.png Debug ===")
print(f"Cropped shape: {cropped.shape}")
print(f"Normalized shape: {norm.shape}")
print(f"Normalized foreground count: {np.count_nonzero(norm == 255)}")
print(f"Normalized unique values: {np.unique(norm)}")

# Check hole detection on normalized image
hf = extract_hole_features(norm, min_hole_area=5)
print(f"\nHole detection (min_hole_area=5):")
print(f"  Total detected: {hf['total_detected_holes']}")
print(f"  Num enclosed: {hf['num_enclosed_holes']}")
print(f"  Hole areas: {hf['hole_areas']}")
print(f"  Hole centers: {hf['hole_centers']}")
print(f"  Hole positions: {hf['normalized_hole_positions']}")

# Check hole detection with very low threshold
hf2 = extract_hole_features(norm, min_hole_area=1)
print(f"\nHole detection (min_hole_area=1):")
print(f"  Total detected: {hf2['total_detected_holes']}")
print(f"  Num enclosed: {hf2['num_enclosed_holes']}")
print(f"  Hole areas: {hf2['hole_areas'][:10]}")

# Check features
nf = extract_normalized_features(cropped, crop_info=info, min_hole_area=5)
print(f"\nNormalized features:")
print(f"  Aspect ratio: {nf['aspect_ratio']:.4f}")
print(f"  Foreground density: {nf['foreground_density']:.4f}")
print(f"  Hole count: {nf['hole_count']}")
print(f"  Hole positions: {nf['normalized_hole_positions']}")
print(f"  Hole areas: {[round(a, 4) for a in nf['normalized_hole_areas']]}")

# Feature classification
cls = classify_features(nf)
print(f"\nFeature classification:")
print(f"  Predicted: {cls['predicted_digit']}")
print(f"  Best score: {cls['best_score']}")
print(f"  Top 5: {[(d, round(s, 4)) for d, s in cls['ranked_candidates'][:5]]}")
print(f"  Observations for digit 0: {cls['rule_observations'][0]}")
print(f"  Observations for digit 3: {cls['rule_observations'][3]}")

# Full recognition
result = recognize_digit(norm, crop_info=info, min_hole_area=20)
print(f"\nFull recognition:")
print(f"  Predicted: {result['predicted_digit']}")
print(f"  Status: {result['status']}")
print(f"  Score: {result['similarity_score']}")
print(f"  Template scores: {[(d, round(s, 4)) for d, s in sorted(result['template_scores'].items(), key=lambda x: -x[1])[:5]]}")
print(f"  Feature scores: {[(d, round(s, 4)) for d, s in sorted(result.get('feature_scores', {}).items(), key=lambda x: -x[1])[:5]]}")
