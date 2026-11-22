import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from preprocessing import preprocess_image, crop_digit, normalize_digit_image
from feature_extraction import extract_all_features, extract_normalized_features
from rule_classifier import classify_features
from template_matcher import match_template
from digit_recognizer import recognize_digit

images_to_test = [
    ('test_images/digit2.png', 2),
    ('test_images/digit2another.png', 2),
    ('test_images/image3.png', 3),
    ('test_images/imagenormal3.png', 3),
]

for img_path, expected_digit in images_to_test:
    print('=' * 80)
    print(f'IMAGE: {img_path} (expected digit: {expected_digit})')
    print('=' * 80)
    
    try:
        image, gray, blurred, binary, cleaned = preprocess_image(img_path)
        print(f'Loaded image shape: {image.shape}')
        print(f'Gray unique values: {np.unique(gray)[:10]}... (total {len(np.unique(gray))})')
        print(f'Binary foreground count: {np.count_nonzero(binary == 255)}')
        print(f'Cleaned foreground count: {np.count_nonzero(cleaned == 255)}')
        
        cropped, info = crop_digit(cleaned, padding=10)
        print(f'Cropped shape: {cropped.shape}, bbox: {info["bbox"]}')
        print(f'Cropped foreground count: {np.count_nonzero(cropped == 255)}')
        
        norm = normalize_digit_image(cropped)
        print(f'Normalized shape: {norm.shape}')
        print(f'Normalized foreground count: {np.count_nonzero(norm == 255)}')
        
        # Template matching
        tmpl = match_template(norm)
        print(f'Template best: digit={tmpl["best_digit"]}, score={tmpl["best_score"]:.4f}')
        print(f'Template top 5: {[(d, round(s, 4)) for d, s in tmpl["ranked_candidates"][:5]]}')
        
        # Feature extraction
        nf = extract_normalized_features(cropped, crop_info=info, min_hole_area=20)
        print(f'Aspect ratio: {nf["aspect_ratio"]:.4f}')
        print(f'Foreground density: {nf["foreground_density"]:.4f}')
        print(f'Hole count: {nf["hole_count"]}')
        print(f'Hole positions: {nf["normalized_hole_positions"]}')
        print(f'Hole areas: {[round(a, 4) for a in nf["normalized_hole_areas"]]}')
        
        # Rule classification
        cls = classify_features(nf)
        print(f'Feature best: digit={cls["predicted_digit"]}, score={cls["best_score"]:.4f}')
        print(f'Feature top 5: {[(d, round(s, 4)) for d, s in cls["ranked_candidates"][:5]]}')
        
        # Full recognition
        result = recognize_digit(norm, crop_info=info, min_hole_area=20)
        print(f'FINAL: pred={result["predicted_digit"]}, status={result["status"]}, score={result["similarity_score"]:.4f}')
        print(f'Expected: {expected_digit}, Correct: {result["predicted_digit"] == expected_digit}')
        
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}')
    
    print()
